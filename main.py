import sys
import time
import torch
import traceback

import utility
import data
import model
import loss
from option import args
from trainer import Trainer


def setup_environment():
    """Initialize deterministic settings, directories, and device configuration."""
    torch.manual_seed(args.seed)
    if not args.cpu and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        device = torch.device("cuda")
        print(f"Using GPU(s): {torch.cuda.device_count()} device(s) available.")
    else:
        device = torch.device("cpu")
        print("Running on CPU mode.")
    return device


def build_pipeline(checkpoint):
    """Build dataset loader, model, and loss function."""
    print("\n Building data loaders...")
    loader = data.Data(args)
    print("Initializing model architecture...")
    sr_model = model.Model(args, checkpoint)
    print("Preparing loss function...")
    loss_fn = loss.Loss(args, checkpoint) if not args.test_only else None
    print("Initialization complete.\n")
    return loader, sr_model, loss_fn


def main():
    # --- Global Safety Nets ---
    try:
        start_time = time.time()
        print("Launching Super-Resolution Training Framework\n" + "═" * 60)
        device = setup_environment()
        checkpoint = utility.checkpoint(args)
        if not checkpoint.ok:
            print("Checkpoint initialization failed.")
            return

        loader, sr_model, loss_fn = build_pipeline(checkpoint)

        print("Launching Trainer...")
        trainer = Trainer(args, loader, sr_model, loss_fn, checkpoint)

        print("\nStarting Training Loop...")
        epoch_counter = 0
        while not trainer.terminate():
            trainer.train()
            trainer.test()
            epoch_counter += 1
            print(f"Completed Epoch {epoch_counter}")

        print("\n Finalizing and Saving Checkpoints...")
        checkpoint.done()

        elapsed = time.time() - start_time
        print(f"\n Training Complete in {elapsed / 3600:.2f} hours ({elapsed:.1f} seconds)")
        print(f" Results saved to: {checkpoint.dir}\n")

    except KeyboardInterrupt:
        print("\n Training interrupted by user. Cleaning up gracefully...")
        try:
            checkpoint.done()
        except Exception:
            pass
        sys.exit(0)

    except Exception as e:
        print("\n FATAL ERROR OCCURRED DURING TRAINING:")
        print("─" * 60)
        traceback.print_exc()
        print("─" * 60)
        try:
            checkpoint.write_log(f"[ERROR] {repr(e)}", refresh=True)
            checkpoint.done()
        except Exception:
            pass
        sys.exit(1)


if __name__ == "__main__":
    main()