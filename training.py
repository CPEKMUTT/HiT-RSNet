"""
-------------------
Customized training and evaluation pipeline for HiT-RSNet and
related super-resolution networks.

This module handles:
- Epoch scheduling and optimizer management
- Data loading and iteration timing
- Logging and checkpointing
- Quantitative evaluation (PSNR, optional result saving)
"""

import os
import math
from decimal import Decimal
from tqdm import tqdm

import torch
import torch.nn.utils as torch_utils
import utility


class HiTTrainer:
    """A high-level training and evaluation controller."""

    def __init__(self, args, data_loader, model, loss_fn, checkpoint):
        self.args = args
        self.scale_factors = args.scale
        self.ckp = checkpoint
        self.model = model
        self.loss_fn = loss_fn
        self.train_loader = data_loader.loader_train
        self.test_loader = data_loader.loader_test

        # Build optimizer and restore if checkpoint exists
        self.optimizer = utility.make_optimizer(args, self.model)
        if args.load != '':
            self.optimizer.load(checkpoint.dir, epoch=len(checkpoint.log))

        self.prev_error = 1e8

    def train_one_epoch(self):
        """Performs one complete training epoch."""
        self.loss_fn.step()  # adjust learning rate if scheduler attached
        epoch = self.optimizer.get_last_epoch() + 1
        lr = self.optimizer.get_lr()

        self.ckp.write_log(f"[Epoch {epoch}]\tLearning rate: {Decimal(lr):.2e}")

        self.loss_fn.start_log()
        self.model.train()

        time_data, time_model = utility.timer(), utility.timer()
        self.train_loader.dataset.set_scale(0)

        for batch_idx, (lr_img, hr_img, _) in enumerate(self.train_loader):
            # Prepare tensors for training
            lr_img, hr_img = self.prepare(lr_img, hr_img)
            time_data.hold()

            # Forward pass
            time_model.tic()
            self.optimizer.zero_grad()

            sr_pred = self.model(lr_img, 0)
            loss = self.loss_fn(sr_pred, hr_img)

            # Backpropagation
            loss.backward()
            if self.args.gclip > 0:
                torch_utils.clip_grad_value_(self.model.parameters(), self.args.gclip)

            self.optimizer.step()
            time_model.hold()

            # Logging
            if (batch_idx + 1) % self.args.print_every == 0:
                self.ckp.write_log(
                    "[{}/{}]\t{}\t{:.1f}+{:.1f}s".format(
                        (batch_idx + 1) * self.args.batch_size,
                        len(self.train_loader.dataset),
                        self.loss_fn.display_loss(batch_idx),
                        time_model.release(),
                        time_data.release(),
                    )
                )
            time_data.tic()

        # End of epoch
        self.loss_fn.end_log(len(self.train_loader))
        self.prev_error = self.loss_fn.log[-1, -1]
        self.optimizer.schedule()

    def evaluate(self):
        """Performs full evaluation across test datasets and scales."""
        torch.set_grad_enabled(False)
        epoch = self.optimizer.get_last_epoch()

        self.ckp.write_log("\nEvaluating model...")
        self.ckp.add_log(torch.zeros(1, len(self.test_loader), len(self.scale_factors)))
        self.model.eval()

        timer_eval = utility.timer()
        if self.args.save_results:
            self.ckp.begin_background()

        for dataset_idx, data_loader in enumerate(self.test_loader):
            for scale_idx, scale in enumerate(self.scale_factors):
                data_loader.dataset.set_scale(scale_idx)
                psnr_total = 0

                for lr_img, hr_img, filename in tqdm(data_loader, ncols=80, desc=f"x{scale}"):
                    lr_img, hr_img = self.prepare(lr_img, hr_img)
                    sr_img = self.model(lr_img, scale_idx)
                    sr_img = utility.quantize(sr_img, self.args.rgb_range)

                    # Compute PSNR
                    self.ckp.log[-1, dataset_idx, scale_idx] += utility.calc_psnr(
                        sr_img, hr_img, scale, self.args.rgb_range, dataset=data_loader
                    )

                    # Save optional outputs
                    if self.args.save_results:
                        save_set = [sr_img]
                        if self.args.save_gt:
                            save_set.extend([lr_img, hr_img])
                        self.ckp.save_results(data_loader, filename[0], save_set, scale)

                # Average over dataset
                self.ckp.log[-1, dataset_idx, scale_idx] /= len(data_loader)
                best = self.ckp.log.max(0)

                self.ckp.write_log(
                    f"[{data_loader.dataset.name} x{scale}]\t"
                    f"PSNR: {self.ckp.log[-1, dataset_idx, scale_idx]:.3f} "
                    f"(Best: {best[0][dataset_idx, scale_idx]:.3f} "
                    f"@Epoch {best[1][dataset_idx, scale_idx] + 1})"
                )

        self.ckp.write_log(f"Forward Time: {timer_eval.toc():.2f}s\nSaving...")
        if self.args.save_results:
            self.ckp.end_background()

        if not self.args.test_only:
            best = self.ckp.log.max(0)
            self.ckp.save(self, epoch, is_best=(best[1][0, 0] + 1 == epoch))

        self.ckp.write_log(f"Total Evaluation Time: {timer_eval.toc():.2f}s\n", refresh=True)
        torch.set_grad_enabled(True)


    # Tensor Preparation Helper

    def prepare(self, *tensors):
        """Moves tensors to the correct device and precision."""
        device = torch.device("cpu" if self.args.cpu else "cuda")

        def _convert(tensor):
            if self.args.precision == "half":
                tensor = tensor.half()
            return tensor.to(device)

        return [_convert(t) for t in tensors]


    def should_terminate(self):
        """Checks whether to stop training based on current epoch."""
        if self.args.test_only:
            self.evaluate()
            return True
        epoch = self.optimizer.get_last_epoch() + 1
        return epoch >= self.args.epochs
