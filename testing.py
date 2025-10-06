import os
import glob
import cv2
import torch
import numpy as np
from tqdm import tqdm

import model
import utility
import common as common
from option import args


device = torch.device("cpu" if args.cpu else "cuda")


def run_inference(args, model):
    """Run HiT-RSNet inference on all images in the input folder."""
    ext = ".png"
    img_paths = sorted(glob.glob(os.path.join(args.dir_data, "*" + ext)))

    if not img_paths:
        raise FileNotFoundError(f"No input images found in {args.dir_data}")

    os.makedirs(args.dir_out, exist_ok=True)

    model.eval()
    print(f"\n[INFO] Running HiT-RSNet inference on {len(img_paths)} image(s)...\n")

    with torch.no_grad():
        for idx, img_path in enumerate(tqdm(img_paths, ncols=80)):
            img_name = os.path.basename(img_path)

            # Load and convert image
            lr_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            lr_img = cv2.cvtColor(lr_img, cv2.COLOR_BGR2RGB)

            # Optional bicubic upscale for low-resolution input
            if args.cubic_input:
                h, w = lr_img.shape[:2]
                new_size = (w * args.scale[0], h * args.scale[0])
                lr_img = cv2.resize(lr_img, new_size, interpolation=cv2.INTER_CUBIC)

            # Convert numpy → tensor
            lr_tensor = common.np2Tensor([lr_img], args.rgb_range)[0].unsqueeze(0).to(device)

            # Choose block-based or full-image inference
            if args.test_block:
                sr_tensor = process_blocks(lr_tensor, model, args)
            else:
                sr_tensor = model(lr_tensor, idx_scale=0)

            # Convert back to numpy and save
            sr_img = tensor_to_image(sr_tensor, args)
            out_path = os.path.join(args.dir_out, img_name)
            cv2.imwrite(out_path, sr_img)

            print(f"[{idx + 1}/{len(img_paths)}] Saved: {out_path}")

    print("\n Inference completed successfully.\n")


def process_blocks(lr, model, args):
    """Process large images block-by-block to avoid GPU memory overflow."""
    b, c, h, w = lr.shape
    scale = args.scale[0]
    patch_out = args.patch_size
    patch_in = patch_out // scale if not args.cubic_input else patch_out

    # Ensure LR input is sufficiently large
    assert h >= patch_in and w >= patch_in, "Input resolution smaller than patch size."

    if not args.cubic_input:
        sr_full = torch.zeros((b, c, h * scale, w * scale), device=lr.device)
    else:
        sr_full = torch.zeros((b, c, h, w), device=lr.device)

    for y in range(0, h, patch_in):
        y = min(y, h - patch_in)
        y_out = scale * y

        for x in range(0, w, patch_in):
            x = min(x, w - patch_in)
            x_out = scale * x

            patch = lr[:, :, y:y + patch_in, x:x + patch_in]
            sr_patch = model(patch)
            sr_full[:, :, y_out:y_out + patch_out, x_out:x_out + patch_out] = sr_patch

    return sr_full

def tensor_to_image(sr_tensor, args):
    """Convert a model output tensor to a displayable image."""
    sr_np = sr_tensor.squeeze(0).cpu().numpy()
    sr_np = np.transpose(sr_np, (1, 2, 0))

    if args.rgb_range == 1:
        sr_np = np.clip(sr_np * 255, 0, 255)
    else:
        sr_np = np.clip(sr_np, 0, args.rgb_range)

    sr_np = sr_np.astype(np.uint8)
    sr_np = cv2.cvtColor(sr_np, cv2.COLOR_RGB2BGR)
    return sr_np

if __name__ == "__main__":
    # Define input, output, and pretrained model paths
    args.pre_train = "../result/our_x2_AID/model/model_best.pt"
    args.dir_data = "../result/AID-dataset/test/LR_x2"
    args.dir_out = "../result/our_x2_AID/result-images"

    # Load model and checkpoint
    checkpoint = utility.checkpoint(args)
    sr_model = model.Model(args, checkpoint)
    sr_model.eval()

    # Run inference
    run_inference(args, sr_model)