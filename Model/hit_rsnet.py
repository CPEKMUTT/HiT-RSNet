from model import common
import torch.nn as nn
from torch.nn import functional as F
import torch
import warnings
from thop import profile, clever_format
from model.hit_rsnet import HiT_RSNet

"""
Author: Naveed Sultan
Date: 2025-10
"""

def make_model(args, parent=False):
    return HiT_RSNet(args)

class CAB(nn.Module):
    """
    Formula:                                     
       Y = σ( f↑( f↓( GAP(X) ) ) ) ⊗ X            
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()

        # Stage ① Global context extraction
        self.context = nn.AdaptiveAvgPool2d(1)       # → (B, C, 1, 1)

        # Stage ② Bottleneck transformation
        r = channels // reduction
        self.bottleneck = nn.Sequential(
            nn.Conv2d(channels, r, kernel_size=1, bias=True),   # ↓ compression
            nn.Conv2d(r, r, kernel_size=1, bias=True),           # inter-feature
            nn.ReLU(inplace=True),
            nn.Conv2d(r, r, kernel_size=1, bias=True),           # pre-expansion
            nn.Conv2d(r, channels, kernel_size=1, bias=True),    # ↑ expansion
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Flow:
            X ──► [AvgPool] ──► [Conv↓→ReLU→Conv↑] ──► A ──► X ⊗ A ──► Y
        """
        # Squeeze spatial context
        y = self.context(x)
        a = self.bottleneck(y)
        # Channel reweighting
        out = x * a
        return out


class RCAM(nn.Module):
    """
    Residual Convolutional Attention Module (RCAM)
    ----------------------------------------

    Architecture:
        [Conv → (BN) → Act] × 4  →  [Convolutional Attention]  →  Residual Add

    Parameters
    ----------
    conv : nn.Module
        Convolution layer constructor (e.g., default_conv).
    n_feat : int
        Number of feature channels.
    kernel_size : int
        Kernel size for each convolution layer.
    reduction : int
        Reduction ratio for channel attention bottleneck.
    bias : bool, optional
        Whether to include bias in convolutions. Default: True.
    bn : bool, optional
        Apply BatchNorm2d if True. Default: False.
    act : nn.Module, optional
        Activation function. Default: nn.ReLU(True).
    res_scale : float, optional
        Scaling factor for residual path. Default: 1.0
    """
    def __init__(
        self, conv, n_feat, kernel_size, reduction,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1.0
    ):
        super().__init__()

        body_layers = []
        for idx in range(4):
            body_layers.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn:
                body_layers.append(nn.BatchNorm2d(n_feat))
            if idx in {1, 3}:
                body_layers.append(act)
        body_layers.append(CAB(n_feat, reduction))

        self.body = nn.Sequential(*body_layers)
        self.res_scale = res_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies sequential convolutions with attention,
        followed by residual feature reinforcement.
        """
        residual = x
        out = self.body(x)
        out = out * self.res_scale + residual
        return out


class PatchEmbedding(nn.Module):
    """
    Args:
        in_channels (int): Number of input channels (e.g., 3 for RGB or n_feats for feature maps).
        embed_dim (int): Dimensionality of the embedding space for each patch token.
        patch_size (int): Spatial resolution of each patch. Default: 4

    Shape:
        Input:  [B, C, H, W]
        Output: [B, N, D] where N = (H / patch_size) × (W / patch_size)
    """
    def __init__(self, in_channels: int, embed_dim: int, patch_size: int = 4):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        # Linear projection from pixel space → embedding space using conv with stride = patch size
        self.projection = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,   # Non-overlapping patches
            bias=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): Input tensor of shape [B, C, H, W]

        Returns:
            Tensor: Embedded patch tokens of shape [B, N, D]
        """
        B, C, H, W = x.shape

        # Patch projection (via conv)
        x = self.projection(x)                      # [B, D, H/ps, W/ps]

        #Flatten spatial dimensions and permute to [B, N, D]
        x = x.flatten(2).transpose(1, 2)            

        return x

class DepthwiseConv(nn.Module):
    """    
    Lightweight replacement for standard convolution,
    commonly used in hierarchical attention and multi-scale feed-forward networks.

    Args:
        in_channels (int): Number of input (and output) channels.
        kernel_size (int): Convolution kernel size. Default: 3
        stride (int): Convolution stride. Default: 1
        padding (int): Zero-padding size. Default: 1
        dilation (int): Dilation rate for expanded receptive field. Default: 1
    """
    def __init__(self,
                 in_channels: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 padding: int = 1,
                 dilation: int = 1):
        super(DepthwiseConv, self).__init__()

        # Depthwise convolution: groups=in_channels ensures independent filtering per channel
        self.depthwise_conv = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,  # critical for depthwise separation
            bias=True
        )

        # Optional normalization for extended architectures (kept disabled for pure HiT-RSNet)
        # self.norm = nn.BatchNorm2d(in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """        
        Args:
            x (Tensor): Input feature map of shape [B, C, H, W].
        
        Returns:
            Tensor: Depthwise convolved output of shape [B, C, H_out, W_out].
        """
        out = self.depthwise_conv(x)
        # Optionally normalize if self.norm is used
        # out = self.norm(out)
        return out



class CWSAB(nn.Module):
    """
    Channel-Wise Self-Attention Block (CWSAB)
    -----------------------------------------
    Implements the channel-wise self-attention mechanism used within the
    Hierarchical Region Transformer Block (HRTB). The block enhances feature
    representation by learning inter-channel dependencies using depth-wise
    convolutions and adaptive gating.

    Structure:
        Q, K, V generation → Channel attention computation → Depth-wise refinement →
        Global adaptive fusion via sigmoid gating.
    """
    def __init__(self, dim: int):
        super(CWSAB, self).__init__()

        # ---- Query, Key, Value Projections (Depth-wise Convolutions) ---- #
        self.query_conv = DepthwiseConv(dim, kernel_size=3, stride=1, padding=1)
        self.key_conv   = DepthwiseConv(dim, kernel_size=3, stride=1, padding=1)
        self.value_conv = DepthwiseConv(dim, kernel_size=3, stride=1, padding=1)

        # ---- Depth-wise Convolution for Local Refinement ---- #
        self.refine_conv = DepthwiseConv(dim, kernel_size=3, stride=1, padding=1)

        # ---- Channel Recalibration Branch ---- #
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_gate = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

        # Normalization (optional future extension) 
        # self.norm = nn.BatchNorm2d(dim)  # can be used for stability

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the CWSAB block.

        Args:
            x (Tensor): Input feature map of shape [B, C, H, W]

        Returns:
            Tensor: Refined feature map with channel-wise attention applied.
        """
        B, C, H, W = x.shape

        # ---- Query, Key, and Value matrices ---- #
        Q = self.query_conv(x).reshape(B, C, -1)           # [B, C, HW]
        K = self.key_conv(x).reshape(B, C, -1)             # [B, C, HW]
        V = self.value_conv(x).reshape(B, C, -1)           # [B, C, HW]

        # ---- Channel-wise Attention Map ---- #
        attn = torch.matmul(Q, K.transpose(-2, -1))        # [B, C, C]
        attn = attn / (C ** 0.5)                           # Scale normalization
        attn = torch.softmax(attn, dim=-1)                 # Channel-level attention weights

        # ---- Weighted Channel Aggregation ---- #
        out = torch.matmul(attn, V).reshape(B, C, H, W)    # Re-map to spatial feature domain

        # ---- Local Spatial Refinement ---- #
        out = self.refine_conv(out)

        # ---- Global Channel Fusion (Squeeze-and-Excitation style) ---- #
        pooled = self.global_pool(out)
        gate = self.channel_gate(pooled)
        out = out * gate                                   # Apply adaptive gating
        return out


# Hierarchical Self Attention Block (HSAB)

class HSAB(nn.Module):
    """
    The HSAB captures both local and long-range dependencies by:
        1. Generating Q, K, V tensors via depth-wise convolutions.
        2. Computing spatial self-attention maps.
        3. Fusing local (1×1), contextual (3×3), and expanded (dilated 3×3) receptive fields.
        4. Producing an adaptive attention mask through a sigmoid gating mechanism.

    This block corresponds to the Hierarchical Self-Attention component
    illustrated in Figure 3(b) of the HiT-RSNet model architecture.

    Args:
        dim (int): Number of feature channels (embedding dimension).
    """
    def __init__(self, dim: int):
        super(HSAB, self).__init__()
        self.query_conv = DepthwiseConv(dim, kernel_size=3, stride=1, padding=1)
        self.key_conv   = DepthwiseConv(dim, kernel_size=3, stride=1, padding=1)
        self.value_conv = DepthwiseConv(dim, kernel_size=3, stride=1, padding=1)

        # Local representation (1×1), contextual (3×3), expanded (dilated 3×3)
        self.local_conv      = nn.Conv2d(dim, dim, kernel_size=1)
        self.context_conv    = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.dilated_fusion  = nn.Conv2d(dim * 2, dim, kernel_size=3, padding=2, dilation=2)

        # Adaptive gating function ---- #
        self.activation_gate = nn.Sigmoid()

        # Optional: normalization layer for stability in deep stacking
        # self.norm = nn.BatchNorm2d(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): Input tensor of shape [B, C, H, W]
        """
        B, C, H, W = x.shape

        # ---- Self-attention computation ---- #
        Q = self.query_conv(x).reshape(B, C, -1).transpose(1, 2)   # [B, HW, C]
        K = self.key_conv(x).reshape(B, C, -1)                     # [B, C, HW]
        V = self.value_conv(x).reshape(B, C, -1).transpose(1, 2)   # [B, HW, C]

        # Scaled dot-product attention
        attention_scores = torch.matmul(Q, K) / (C ** 0.5)
        attention_probs = torch.softmax(attention_scores, dim=-1)
        attn_out = torch.matmul(attention_probs, V).transpose(1, 2).reshape(B, C, H, W)
        local_feat = self.local_conv(attn_out)
        context_feat = self.context_conv(attn_out)

        # Concatenate features for multi-receptive aggregation
        fused_features = torch.cat([local_feat, context_feat], dim=1)
        attn_mask = self.activation_gate(self.dilated_fusion(fused_features))
        out = attn_out * attn_mask

        # out = self.norm(out) + x (optional residual norm)

        return out

# Attention-Guided Fusion (AGF)
class AGF(nn.Module):

    def __init__(self, dim: int):
        super(AGF, self).__init__()

        # ---- Gating Convolution ---- #
        #  concatenate features [Fc, Fh] → produces scalar attention map A
        self.gate_conv = nn.Conv2d(
            in_channels = dim * 2,
            out_channels = dim,
            kernel_size = 1,
            stride = 1,
            padding = 0,
            bias = True
        )
        # Sigmoid ensures gating values between [0, 1]
        self.activation = nn.Sigmoid()

def forward(self, Fc: torch.Tensor, Fh: torch.Tensor) -> torch.Tensor:
        """
        Args:
            Fc (Tensor): Feature map from Channel-Wise Self-Attention [B, C, H, W].
            Fh (Tensor): Feature map from Hierarchical Self-Attention [B, C, H, W].
        """
        # ---- input feature streams 
        fused_features = torch.cat([Fc, Fh], dim=1)          # [B, 2C, H, W]
        attention_mask = self.activation(self.gate_conv(fused_features))  # [B, C, H, W]  # ---- fusion attention mask ---- #
        fused_output = Fc * attention_mask + Fh * (1 - attention_mask) # ---- Weighted fusion (adaptive feature blending) ---- #
        return fused_output

# Multi-Layer Feed-Forward Block (MLFFB)

class MLFFB(nn.Module):
    """
    Functional Overview:
        1. Layer normalization in feature space.
        2. Parallel multi-scale convolutional expansion.
        3. Channel splitting and non-linear activation across scales.
        4. Hierarchical recombination (cross-kernel interaction).
        5. Final spatial consolidation and dropout regularization.

    Args:
        dim (int): Number of input channels.
        hidden_dim (int): Number of intermediate (expanded) channels.
        dropout (float): Dropout probability. Default: 0.0
    """
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super(MLFFB, self).__init__()
        self.norm = nn.LayerNorm(dim)

        # ---- Parallel Depthwise Convolutional Expansion ---- #
        self.expand_3x3 = nn.Conv2d(dim, hidden_dim, kernel_size=3, padding=1, groups=1)
        self.expand_5x5 = nn.Conv2d(dim, hidden_dim, kernel_size=5, padding=2, groups=1)
        self.expand_7x7 = nn.Conv2d(dim, hidden_dim, kernel_size=7, padding=3, groups=1)

        # ---- Depthwise Refinement (Grouped Convs per Branch) ---- #
        self.refine_3x3 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim)
        self.refine_5x5 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=5, padding=2, groups=hidden_dim)
        self.refine_7x7 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=7, padding=3, groups=hidden_dim)

        self.fuse_projection = nn.Conv2d(hidden_dim * 3, dim, kernel_size=1) # ---- Final Fusion Projection

        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): Input tensor of shape [B, C, H, W].
        """
        B, C, H, W = x.shape
        x_norm = x.permute(0, 2, 3, 1).contiguous()     # [B, C, H, W] → [B, H, W, C]
        x_norm = self.norm(x_norm)
        x_norm = x_norm.permute(0, 3, 1, 2).contiguous() # [B, H, W, C] → [B, C, H, W]

        # ---- Multi-Scale Expansion ---- #
        feat3 = self.expand_3x3(x_norm)
        feat5 = self.expand_5x5(x_norm)
        feat7 = self.expand_7x7(x_norm)

        # ---- Channel Segmentation & Nonlinearity ---- #
        seg3 = [self.act(chunk) for chunk in torch.chunk(feat3, 3, dim=1)]
        seg5 = [self.act(chunk) for chunk in torch.chunk(feat5, 3, dim=1)]
        seg7 = [self.act(chunk) for chunk in torch.chunk(feat7, 3, dim=1)]

        # ---- Cross-Scale Channel Intermixing ---- #
        mixed3 = torch.cat([seg3[0], seg5[1], seg7[2]], dim=1)
        mixed5 = torch.cat([seg5[0], seg7[1], seg3[2]], dim=1)
        mixed7 = torch.cat([seg7[0], seg3[1], seg5[2]], dim=1)

        # ---- Hierarchical Refinement ---- #
        ref3 = self.act(self.refine_3x3(mixed3))
        ref5 = self.act(self.refine_5x5(mixed5))
        ref7 = self.act(self.refine_7x7(mixed7))
        fused = torch.cat([ref3, ref5, ref7], dim=1)  # ---- Final Fusion ---- #
        output = self.fuse_projection(fused)
        return self.drop(output)


class HRTBUnit(nn.Module):
    """
    HRTBUnit
    --------
    A single Hierarchical Region Transformer Block (HRTB) unit.

    Combines:
        • CWSAB → Channel-Wise Self-Attention (global dependency)
        • HSAB  → Hierarchical Self-Attention (local & multi-scale)
        • AGF   → Attention-Guided Fusion (adaptive merging)
        • MLFFB → Multi-Layer Feed-Forward Block (refinement)

    Overall:
        x' = MLFFB( AGF( CWSAB(x), HSAB(x) ) ) + x
    """
    def __init__(self, dim: int):
        super().__init__()

        # --- Sub-Modules --- #
        self.channel_attention = CWSAB(dim)
        self.hierarchical_attention = HSAB(dim)
        self.fusion = AGF(dim)
        self.feed_forward = MLFFB(dim, hidden_dim=dim * 3)

        # Optional residual scaling (useful in deeper stacks)
        self.res_scale = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for one HRTB unit.
        """
        Fc = self.channel_attention(x)
        Fh = self.hierarchical_attention(x)
        Ffusion = self.fusion(Fc, Fh)
        Fout = self.feed_forward(Ffusion)
        return x + self.res_scale * Fout



class HRTB(nn.Module):
    def __init__(self, in_channels=64, embed_dim=64, depth=3, patch_size=4, img_size=48):
        """
        Multi-stage transformer operating on patch embeddings.
        Each stage refines spatial semantics through stacked HRTBUnits.
        """
        super().__init__()

        # ---- Patch & Positional Encoding ---- #
        self.patch_embed = PatchEmbedding(in_channels, embed_dim, patch_size)
        num_patches = (img_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, embed_dim))

        # ---- Core Transformer Layers ---- #
        self.layers = nn.ModuleList([HRTBUnit(embed_dim) for _ in range(depth)])
        self.proj_back = nn.Conv2d(embed_dim, in_channels, kernel_size=1) # ---- Reconstruction 

        self.patch_size = patch_size # Meta info
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x = self.patch_embed(x)                  # [B, N, D]
        if x.size(1) != self.pos_embed.size(1):  # Dynamic positional embedding
            self.pos_embed = nn.Parameter(
                torch.randn(1, x.size(1), x.size(2), device=x.device)
            )
        x = x + self.pos_embed
        x = x.transpose(1, 2).reshape(B, self.embed_dim, H // self.patch_size, W // self.patch_size) # Tokens → spatial map
        for blk in self.layers: # Pass through stacked transformer units
            x = blk(x)

        # Upsample back to original resolution
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
        return self.proj_back(x)




class HiT_RSNet(nn.Module):
    """
    HiT-RSNet: Hybrid Transformer-Convolutional Residual Network
    ------------------------------------------------------------
    Purpose:
        Multi-branch super-resolution model for remote sensing imagery.
        Combines Residual Convolutional Attention Modules (RCAM) with a 
        Hierarchical Residual Transformer Block (HRTB) and fusion gating.

    Architecture Flow:
        Input → MeanShift → Head Conv
              → [RCAM Stack] → [HRTB]
              → Fusion(Concat+Conv1×1)
              → Upsampler + Reconstruction
              → MeanShift (restore)
              → Output
    """

    def __init__(self, args, conv=common.default_conv):
        super().__init__()

        # ----- Core Settings -----
        self.scale = args.scale[0]
        self.rgb_range = args.rgb_range
        self.num_blocks = args.n_resgroups
        self.feat_channels = args.n_feats
        self.reduction_ratio = args.reduction
        self.act_fn = nn.ReLU(inplace=True)

        #  Mean Normalization
        self.sub_mean = common.MeanShift(self.rgb_range)
        self.head = self._build_head(conv)
        self.body = self._build_body(conv)
        self.hrtb_branch = HRTB(in_channels=self.feat_channels)
        self.fusion = nn.Conv2d(self.feat_channels * 2, self.feat_channels, kernel_size=1)
        self.tail = self._build_tail(conv)
        self.add_mean = common.MeanShift(self.rgb_range, sign=1) # ----- Mean Shift Back -----

    def _build_head(self, conv):
        """Head: basic feature extraction"""
        return nn.Sequential(
            conv(in_channels=3, out_channels=self.feat_channels, kernel_size=3)
        )

    def _build_body(self, conv):
        """Body: stack of RCAM modules followed by a closing conv"""
        modules = [
            RCAM(conv, self.feat_channels, 3, self.reduction_ratio,
                 bias=True, bn=False, act=self.act_fn, res_scale=1.0)
            for _ in range(self.num_blocks)
        ]
        modules.append(conv(self.feat_channels, self.feat_channels, kernel_size=3))
        return nn.Sequential(*modules)

    def _build_tail(self, conv):
        """Tail: upsampling and reconstruction"""
        return nn.Sequential(
            common.Upsampler(conv, self.scale, self.feat_channels, act=False),
            conv(self.feat_channels, 3, kernel_size=3)
        )

    def forward(self, x):
        """
        Forward flow:
            Input → SubMean → Head → [Body + HRTB]
            → Fusion + Skip → Tail → AddMean → Output
        """
        # --- Preprocess ---
        x = self.sub_mean(x)
        feat = self.head(x)

        # --- Dual Feature Streams ---
        feat_rcam = self.body(feat)
        feat_hrt = self.hrtb_branch(feat)

        # --- Feature Fusion ---
        combined = torch.cat((feat_rcam, feat_hrt), dim=1)
        fused = self.fusion(combined)
        fused = fused + feat  # residual reinforcement
        out = self.tail(fused) # --- Reconstruction ---
        out = self.add_mean(out)
        return out


    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))


class ModelConfig:
    def __init__(self):
        self.n_resgroups = 4          # Number of RCAM blocks
        self.n_feats = 64             # Feature channels
        self.reduction = 8            # Channel reduction ratio
        self.scale = [2]              # Upscaling factor
        self.rgb_range = 255
        self.n_colors = 3
        self.model = "HiT_RSNet"

def initialize_model(device="cuda"):
    args = ModelConfig()
    model = HiT_RSNet(args)
    if torch.cuda.is_available() and device == "cuda":
        model = model.cuda()
    return model

def compute_complexity(model, input_size=(1, 3, 48, 48)):

    # Synthetic input tensor
    dummy_input = torch.randn(*input_size).cuda()
    def _noop_count(m, x, y):
        m.total_ops = torch.zeros(1)

    custom_hooks = {
        "MeanShift": _noop_count,
        "Upsampler": _noop_count,
        "PixelShuffle": _noop_count,
        "Sequential": _noop_count,
        "Sigmoid": _noop_count,
    }

    # Patch FLOP handlers into THOP registry (defensive)
    for layer, hook_fn in custom_hooks.items():
        if hasattr(model, layer):
            setattr(model, layer, hook_fn)

    # Perform profiling
    flops, params = profile(model, inputs=(dummy_input,), verbose=False)
    flops, params = clever_format([flops, params], "%.3f")

    return flops, params
if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning, module="torch.optim.lr_scheduler")

    model = initialize_model()

    total_flops, total_params = compute_complexity(model)

    print("\n───────────────────────────────")
    print(" HiT-RSNet Complexity Summary")
    print("───────────────────────────────")
    print(f"Total Parameters : {total_params}")
    print(f"Total FLOPs (G)  : {total_flops}")
    print("───────────────────────────────")


