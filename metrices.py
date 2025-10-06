import os
import math
import cv2
import glob
import numpy as np
import concurrent.futures
from sewar.full_ref import sam, scc


class ImageMetrics:
    """Encapsulates image quality metrics for SR evaluation."""

    @staticmethod
    def psnr(img1, img2):
        img1, img2 = img1.astype(np.float64), img2.astype(np.float64)
        mse = np.mean((img1 - img2) ** 2)
        if mse == 0:
            return float("inf")
        return 20 * math.log10(255.0 / math.sqrt(mse))

    @staticmethod
    def rgb_psnr(img1, img2):
        """Compute PSNR across RGB channels and average."""
        if img1.ndim == 3:
            return np.mean([ImageMetrics.psnr(img1[..., c], img2[..., c]) for c in range(3)])
        return ImageMetrics.psnr(img1, img2)

    @staticmethod
    def ssim(img1, img2):
        """Compute SSIM for grayscale or RGB."""
        if img1.ndim == 3:
            return np.mean([ImageMetrics._single_ssim(img1[..., c], img2[..., c]) for c in range(3)])
        return ImageMetrics._single_ssim(img1, img2)

    @staticmethod
    def _single_ssim(img1, img2):
        """Helper for single-channel SSIM calculation."""
        C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
        img1, img2 = img1.astype(np.float64), img2.astype(np.float64)

        window = np.outer(
            cv2.getGaussianKernel(11, 1.5),
            cv2.getGaussianKernel(11, 1.5).T
        )

        mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
        mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
        sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1 ** 2
        sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2 ** 2
        sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1 * mu2

        ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / (
            (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2)
        )
        return ssim_map.mean()

    @staticmethod
    def spectral_angle_mapper(gt, gen):
        """Wrapper for SAM (Spectral Angle Mapper)."""
        return float(sam(gt, gen))

    @staticmethod
    def spatial_correlation_coefficient(gt, gen):
        """Wrapper for SCC (Spatial Correlation Coefficient)."""
        return float(scc(gt, gen))


def crop_border(img, border):
    if border == 0:
        return img
    if img.ndim == 3:
        return img[border:-border, border:-border, :]
    return img[border:-border, border:-border]


def to_y_channel(img):
    """Convert BGR image to Y channel of YCbCr."""
    img = img.astype(np.float32)
    return np.dot(img, [24.966, 128.553, 65.481]) / 255.0 + 16.0


def load_image(path, normalize=True):
    """Load image with OpenCV and convert to RGB float32."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if normalize:
        img = img / 255.0
    return img


class SREvaluator:
    def __init__(self, gt_folder, gen_folder, suffix="_x4_SR", crop_border=4, test_Y=False):
        self.gt_folder = gt_folder
        self.gen_folder = gen_folder
        self.suffix = suffix
        self.crop_border = crop_border
        self.test_Y = test_Y
        self.metrics = ImageMetrics()

        self.results = {"PSNR": [], "SSIM": [], "SAM": [], "SCC": []}

    # ----------------------------------------------------------------------
    def evaluate_image_pair(self, gt_path):
        """Evaluate a single pair of GT and generated image."""
        base = os.path.splitext(os.path.basename(gt_path))[0]
        gen_path = os.path.join(self.gen_folder, base + self.suffix + ".png")

        gt = load_image(gt_path)
        gen = load_image(gen_path)

        if self.test_Y and gt.ndim == 3:
            gt, gen = to_y_channel(gt), to_y_channel(gen)

        gt_cropped = crop_border(gt, self.crop_border)
        gen_cropped = crop_border(gen, self.crop_border)

        gt_scaled = (gt_cropped * 255).astype(np.float64)
        gen_scaled = (gen_cropped * 255).astype(np.float64)

        psnr = self.metrics.rgb_psnr(gt_scaled, gen_scaled)
        ssim = self.metrics.ssim(gt_scaled, gen_scaled)
        sam_val = self.metrics.spectral_angle_mapper(gt_scaled, gen_scaled)
        scc_val = self.metrics.spatial_correlation_coefficient(gt_scaled, gen_scaled)

        return base, psnr, ssim, sam_val, scc_val

    # ----------------------------------------------------------------------
    def run_evaluation(self, workers=4):
        gt_images = sorted(glob.glob(os.path.join(self.gt_folder, "*")))
        if not gt_images:
            raise ValueError(f"No images found in {self.gt_folder}")

        print(f"Evaluating {len(gt_images)} images with {workers} threads...\n")

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(self.evaluate_image_pair, p): p for p in gt_images}
            for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
                try:
                    base, psnr, ssim, sam_val, scc_val = future.result()
                    print(f"[{i:03d}] {base:25s} | PSNR: {psnr:6.3f} | SSIM: {ssim:6.4f} | "
                          f"SAM: {sam_val:6.4f} | SCC: {scc_val:6.4f}")

                    self.results["PSNR"].append(psnr)
                    self.results["SSIM"].append(ssim)
                    self.results["SAM"].append(sam_val)
                    self.results["SCC"].append(scc_val)
                except Exception as e:
                    print(f"[ERROR] Skipping {futures[future]} due to: {e}")

        self._print_summary()

    # ----------------------------------------------------------------------
    def _print_summary(self):
        """Print average results with clear formatting."""
        avg = {k: np.mean(v) for k, v in self.results.items()}
        print("\n" + "=" * 65)
        print("FINAL AVERAGE METRICS")
        print("=" * 65)
        print(f"PSNR: {avg['PSNR']:.4f} dB")
        print(f"SSIM: {avg['SSIM']:.4f}")
        print(f"SAM:  {avg['SAM']:.4f}°")
        print(f"SCC:  {avg['SCC']:.4f}")
        print("=" * 65 + "\n")

if __name__ == "__main__":
    gt_dir = "../result/x4_RSSCN7/HR"
    sr_dir = "../result/x4_RSSCN7/results"

    evaluator = SREvaluator(
        gt_folder=gt_dir,
        gen_folder=sr_dir,
        suffix="_x4_SR",
        crop_border=4,
        test_Y=False
    )
    evaluator.run_evaluation(workers=8)