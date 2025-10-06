import os
import math
import time
import datetime
import threading
from multiprocessing import Process, Queue

import numpy as np
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio

class Timer:
    """High-precision timer with accumulation and checkpointing support."""

    def __init__(self):
        self.reset()
        self.start()

    def start(self):
        """Start or restart the timer."""
        self._t0 = time.time()

    def stop(self, restart=False):
        """Return elapsed time. Optionally restart timer."""
        elapsed = time.time() - self._t0
        if restart:
            self._t0 = time.time()
        return elapsed

    def hold(self):
        """Accumulate elapsed time."""
        self._acc += self.stop()

    def release(self):
        """Return accumulated time and reset counter."""
        total = self._acc
        self._acc = 0.0
        return total

    def reset(self):
        """Reset accumulated time."""
        self._acc = 0.0


class Checkpoint:


    def __init__(self, args):
        self.args = args
        self.success = True
        self.log_tensor = torch.Tensor()

        # ---------------- Directory Setup ----------------
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        base_dir = args.load or args.save or timestamp
        self.dir = os.path.join("./result", base_dir)

        # Reset experiment folder if requested
        if args.reset and os.path.exists(self.dir):
            print(f" Resetting experiment folder: {self.dir}")
            os.system(f"rm -rf {self.dir}")

        os.makedirs(self.dir, exist_ok=True)
        os.makedirs(self.get_path("model"), exist_ok=True)
        for dataset in args.data_test:
            os.makedirs(self.get_path(f"results-{dataset}"), exist_ok=True)

        # ---------------- Log Initialization ----------------
        self._init_logs(timestamp)
        self.n_proc = 8  # background save workers

        # Resume from existing log if found
        if args.load and os.path.exists(self.get_path("psnr_log.pt")):
            self.log_tensor = torch.load(self.get_path("psnr_log.pt"))
            print(f" Continuing training from epoch {len(self.log_tensor)}")

    def _init_logs(self, timestamp):
        """Initialize text logs and configuration snapshot."""
        self.log_file = open(self.get_path("log.txt"), "a", buffering=1)
        with open(self.get_path("config.txt"), "w") as cfg:
            cfg.write(f"Experiment Time: {timestamp}\n\n")
            for key, val in vars(self.args).items():
                cfg.write(f"{key:20s}: {val}\n")

    def get_path(self, *paths):
        return os.path.join(self.dir, *paths)

    def save_checkpoint(self, trainer, epoch, best=False):
        """Save model weights, optimizer state, and metrics."""
        trainer.model.save(self.get_path("model"), epoch, is_best=best)
        trainer.loss.save(self.dir)
        trainer.loss.plot_loss(self.dir, epoch)

        self._plot_psnr(epoch)
        trainer.optimizer.save(self.dir)
        torch.save(self.log_tensor, self.get_path("psnr_log.pt"))

    def add_log(self, new_log):
        """Append new epoch log tensor."""
        self.log_tensor = torch.cat([self.log_tensor, new_log])

    def write_log(self, text, refresh=False):
        """Write console and file logs."""
        print(text)
        self.log_file.write(text + "\n")
        if refresh:
            self.log_file.flush()

    def finish(self):
        """Close log files cleanly."""
        if self.log_file:
            self.log_file.close()

    def _plot_psnr(self, epoch):
        """Plot PSNR over epochs per dataset and scale."""
        axis = np.arange(1, epoch + 1)
        for i, dataset in enumerate(self.args.data_test):
            plt.figure(figsize=(6, 4))
            for j, scale in enumerate(self.args.scale):
                plt.plot(
                    axis,
                    self.log_tensor[:, i, j].numpy(),
                    label=f"×{scale}"
                )
            plt.title(f"PSNR on {dataset}")
            plt.xlabel("Epoch")
            plt.ylabel("PSNR (dB)")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(self.get_path(f"PSNR_{dataset}.pdf"))
            plt.close()

    def begin_background_saver(self):
        """Launch asynchronous threads for saving output images."""
        self.queue = Queue()

        def writer_task(q):
            while True:
                job = q.get()
                if job is None:
                    break
                filename, img_tensor = job
                imageio.imwrite(filename, img_tensor.numpy())

        self.workers = [
            Process(target=writer_task, args=(self.queue,)) for _ in range(self.n_proc)
        ]
        for w in self.workers:
            w.start()

    def end_background_saver(self):
        """Cleanly terminate all writer processes."""
        for _ in range(self.n_proc):
            self.queue.put(None)
        for w in self.workers:
            w.join()

    def save_results(self, dataset, filename, tensor_list, scale):
        """Push images into the asynchronous writer queue."""
        if not self.args.save_results:
            return
        prefix = f"{filename}_x{scale}_"
        save_dir = self.get_path(f"results-{dataset.dataset.name}")
        postfix = ("SR", "LR", "HR")
        for img, tag in zip(tensor_list, postfix):
            normalized = img[0].mul(255 / self.args.rgb_range)
            tensor_cpu = normalized.byte().permute(1, 2, 0).cpu()
            save_path = os.path.join(save_dir, f"{prefix}{tag}.png")
            self.queue.put((save_path, tensor_cpu))


def quantize(img, rgb_range):
    """Clamp and rescale image to valid pixel range."""
    scale_factor = 255 / rgb_range
    return img.mul(scale_factor).clamp(0, 255).round().div(scale_factor)


def calc_psnr(sr, hr, scale, rgb_range, dataset=None):
    """Compute PSNR with luminance adjustment for benchmark datasets."""
    if hr.numel() == 1:
        return 0.0

    diff = (sr - hr) / rgb_range
    shave = scale if (dataset and dataset.dataset.benchmark) else scale + 6

    if diff.size(1) > 1 and dataset and dataset.dataset.benchmark:
        weights = torch.tensor([65.738, 129.057, 25.064], device=diff.device) / 256
        diff = diff.mul(weights.view(1, 3, 1, 1)).sum(dim=1)

    valid = diff[..., shave:-shave, shave:-shave]
    mse = valid.pow(2).mean().item()
    return float("inf") if mse == 0 else -10 * math.log10(mse)


def build_optimizer(args, model):
    """
    Create optimizer with custom scheduler.
    Supports: SGD, Adam, RMSprop, and flexible decay strategies.
    """
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    opt_kwargs = {"lr": args.lr, "weight_decay": args.weight_decay}

    # Select optimizer
    if args.optimizer == "SGD":
        optimizer_class = optim.SGD
        opt_kwargs["momentum"] = args.momentum
    elif args.optimizer == "ADAM":
        optimizer_class = optim.Adam
        opt_kwargs["betas"], opt_kwargs["eps"] = args.betas, args.epsilon
    elif args.optimizer == "RMSprop":
        optimizer_class = optim.RMSprop
        opt_kwargs["eps"] = args.epsilon
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")

    # Learning rate scheduler
    milestones = [int(x) for x in args.decay.split("-")]
    scheduler_kwargs = {"milestones": milestones, "gamma": args.gamma}
    scheduler_class = lrs.MultiStepLR

    class ManagedOptimizer(optimizer_class):
        """Optimizer wrapper with integrated scheduler tracking."""

        def __init__(self, *o_args, **o_kwargs):
            super().__init__(*o_args, **o_kwargs)
            self.scheduler = scheduler_class(self, **scheduler_kwargs)

        def save(self, directory):
            torch.save(self.state_dict(), os.path.join(directory, "optimizer.pt"))

        def load(self, directory, epoch=1):
            self.load_state_dict(torch.load(os.path.join(directory, "optimizer.pt")))
            for _ in range(epoch):
                self.scheduler.step()

        def step_scheduler(self):
            self.scheduler.step()

        def current_lr(self):
            return self.scheduler.get_last_lr()[0]

        def last_epoch(self):
            return self.scheduler.last_epoch

    return ManagedOptimizer(trainable_params, **opt_kwargs)

