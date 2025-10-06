import os
import torch
import torch.nn as nn
import torch.nn.parallel as parallel
from importlib import import_module

class ModelWrapper(nn.Module):
    """
    ModelWrapper: Unified Interface for HiT-RSNet and Related Architectures
    -----------------------------------------------------------------------
    Handles dynamic model loading, precision management, checkpointing,
    and distributed/multi-GPU execution.

    Attributes
    ----------
    model : nn.Module
        The loaded network architecture.
    device : torch.device
        Computation device (CPU or CUDA).
    n_GPUs : int
        Number of available GPUs.
    save_models : bool
        Whether to save model checkpoints per epoch.
    """

    def __init__(self, args, checkpoint):
        super().__init__()

        # --- Basic Configuration ---
        self.scale = args.scale
        self.current_scale = 0
        self.device = torch.device("cpu" if args.cpu else "cuda")
        self.precision = args.precision
        self.cpu_mode = args.cpu
        self.multi_gpu = args.n_GPUs > 1
        self.save_models = args.save_models
        self.n_GPUs = args.n_GPUs

        # --- Dynamically Import Target Model ---
        model_module = import_module(f"model.{args.model.lower()}")
        self.model = model_module.make_model(args).to(self.device)

        if self.precision == "half":
            self.model = self.model.half()

        # --- Load Pretrained or Resumed Weights ---
        self._load_checkpoint(
            path=checkpoint.get_path("model"),
            pretrain=args.pre_train,
            resume=args.resume,
            cpu=args.cpu
        )

        # Log the architecture summary
        print(self.model, file=checkpoint.log_file)

    def forward(self, x, scale_idx):
        """
        Forward propagation wrapper with multi-GPU support and dynamic scaling.
        """
        self.current_scale = scale_idx
        if hasattr(self.model, "set_scale"):
            self.model.set_scale(scale_idx)

        if self.training and self.multi_gpu:
            return parallel.data_parallel(self.model, x, range(self.n_GPUs))
        else:
            return self.model(x)

    def save(self, directory, epoch, is_best=False):
        """
        Saves model state to disk with tagging for 'latest', 'best', and per-epoch snapshots.
        """
        checkpoints = [os.path.join(directory, "model_latest.pt")]

        if is_best:
            checkpoints.append(os.path.join(directory, "model_best.pt"))
        if self.save_models:
            checkpoints.append(os.path.join(directory, f"model_epoch_{epoch}.pt"))

        state_dict = self.model.state_dict()
        for path in checkpoints:
            torch.save(state_dict, path)


    def _load_checkpoint(self, path, pretrain="", resume=-1, cpu=False):
        """
        Loads model weights based on pretraining and resume mode.
        - resume = -1 : load latest checkpoint
        - resume = 0  : load from pretrained path or download
        - resume > 0  : load specific epoch checkpoint
        """
        kwargs = {"map_location": "cpu"} if cpu else {}
        model_state = None

        if resume == -1:
            ckpt_path = os.path.join(path, "model_latest.pt")
            if os.path.exists(ckpt_path):
                model_state = torch.load(ckpt_path, **kwargs)

        elif resume == 0:
            if pretrain == "download":
                print("Downloading pretrained model...")
                model_state = torch.utils.model_zoo.load_url(
                    self.model.url,
                    model_dir=os.path.join("..", "models"),
                    **kwargs
                )
            elif pretrain and os.path.exists(pretrain):
                print(f"Loading pretrained model from: {pretrain}")
                model_state = torch.load(pretrain, **kwargs)
            else:
                print("No valid pretrained model found.")

        else:  # resume from a specific epoch
            ckpt_path = os.path.join(path, f"model_epoch_{resume}.pt")
            if os.path.exists(ckpt_path):
                model_state = torch.load(ckpt_path, **kwargs)
            else:
                print(f"Epoch checkpoint not found: {ckpt_path}")

        # Apply loaded weights
        if model_state is not None:
            self.model.load_state_dict(model_state, strict=False)
        else:
            print(" No weights loaded (training from scratch).")



