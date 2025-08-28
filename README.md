# HiT-RSNet
This work introduces HiT-RSNet, a hybrid transformer–convolutional super-resolution framework tailored for remote sensing imagery. Existing transformer-based SR methods often fail to preserve fine textures, sharp boundaries, and spectral–spatial consistency due to limited global modeling and weak local refinement.

HiT-RSNet addresses these issues through a dual-branch design:

1- Hierarchical Region Transformer Blocks (HRTB): Capture both local and global dependencies via channel-wise attention (CWSAB), hierarchical spatial attention (HSAB), and multi-layer feed-forward blocks (MLFFB).

2- Residual Convolutional Attention Modules (RCAM): Refine local structures with convolutional attention, enabling robust texture recovery across varied spatial scales.

3- Cross-Path Fusion (CPF): Integrates transformer and convolutional features adaptively for balanced global–local representation.

4- Multi-stage feature refinement: Ensures progressive enhancement of intermediate layers, reducing noise and preserving fine details.

The model is evaluated on four benchmark datasets (UCMerced, AID, RSCNN7, WHU-RS19) under ×2, ×3, and ×4 upscaling factors. Results consistently show state-of-the-art performance, with HiT-RSNet surpassing CNN-based (VDSR, HSENet) and transformer-based (TransENet, SwinIR) baselines in PSNR, SSIM, SCC, and SAM metrics.

Overall, HiT-RSNet demonstrates the importance of hybrid architectures for remote sensing SR, achieving sharper boundaries, improved edge alignment, and better spectral consistency, which are crucial for tasks like urban monitoring, land-use mapping, and disaster response.
