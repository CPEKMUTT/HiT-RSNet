import os
import glob
import pickle
import random
import imageio
from torch.utils.data import Dataset
from data import common


class SRData(Dataset):

    def __init__(self, args, name: str = '', train: bool = True, benchmark: bool = False):
        super().__init__()

        # ---------------- Core Parameters ----------------
        self.args = args
        self.name = name
        self.is_train = train
        self.split = "train" if train else "test"
        self.benchmark = benchmark
        self.scale_factors = args.scale
        self.current_scale_index = 0
        self.large_input = args.model.upper() == "VDSR"
        self.do_eval = not train

        # ---------------- Path Setup ----------------
        self._initialize_paths(args.dir_data)

        # ---------------- File List Scanning ----------------
        if args.ext.find("img") < 0:
            bin_dir = os.path.join(self.root_path, "bin")
            os.makedirs(bin_dir, exist_ok=True)

        self.hr_list, self.lr_list = self._collect_image_paths()
        self._prepare_file_lists(args.ext)

        # ---------------- Repeat Count ----------------
        if self.is_train:
            num_patches = args.batch_size * args.test_every
            num_images = len(args.data_train) * len(self.hr_list)
            self.repeat_factor = max(num_patches // max(num_images, 1), 1)
        else:
            self.repeat_factor = 1


    def _initialize_paths(self, data_dir: str):
        """Define HR/LR data directory structure."""
        self.root_path = os.path.join(data_dir, self.name)
        self.hr_dir = os.path.join(self.root_path, "bin", "HR")
        self.lr_dir = os.path.join(self.root_path, "bin", "LR_bicubic")

        if self.large_input:
            self.lr_dir += "L"  # Adjust for large-input models

        self.file_ext = (".png", ".png")

    def _collect_image_paths(self):
        """Collect HR and LR file paths for all scales."""
        hr_files = sorted(glob.glob(os.path.join(self.hr_dir, "*" + self.file_ext[0])))
        lr_files_per_scale = [[] for _ in self.scale_factors]

        for hr_file in hr_files:
            base_name, _ = os.path.splitext(os.path.basename(hr_file))
            for scale_idx, scale_val in enumerate(self.scale_factors):
                lr_files_per_scale[scale_idx].append(
                    os.path.join(
                        self.lr_dir, f"X{scale_val}/{base_name}x{scale_val}{self.file_ext[1]}"
                    )
                )

        return hr_files, lr_files_per_scale


    def _prepare_file_lists(self, ext_type: str):
        """Prepare image lists and generate .pt cache if required."""
        if ext_type.find("img") >= 0 or self.benchmark:
            self.images_hr, self.images_lr = self.hr_list, self.lr_list
            return

        bin_root = os.path.join(self.root_path, "bin")
        os.makedirs(os.path.join(bin_root, "HR"), exist_ok=True)
        for scale in self.scale_factors:
            os.makedirs(os.path.join(bin_root, f"LR_bicubic/X{scale}"), exist_ok=True)

        self.images_hr, self.images_lr = [], [[] for _ in self.scale_factors]

        for hr_path in self.hr_list:
            bin_path = hr_path.replace(self.root_path, bin_root).replace(self.file_ext[0], ".pt")
            self.images_hr.append(bin_path)
            self._create_binary_cache(ext_type, hr_path, bin_path)

        for scale_idx, lr_paths in enumerate(self.lr_list):
            for lr_path in lr_paths:
                bin_path = lr_path.replace(self.root_path, bin_root).replace(self.file_ext[1], ".pt")
                self.images_lr[scale_idx].append(bin_path)
                self._create_binary_cache(ext_type, lr_path, bin_path)

    def _create_binary_cache(self, ext_type, img_path, bin_path):
        """Convert image to cached tensor file if not found."""
        if not os.path.isfile(bin_path) or "reset" in ext_type:
            print(f"⚙️  Creating binary cache: {bin_path}")
            with open(bin_path, "wb") as f:
                pickle.dump(imageio.imread(img_path), f)


    def __getitem__(self, index):
        lr, hr, fname = self._read_pair(index)
        lr, hr = self._prepare_patch(lr, hr)
        lr, hr = common.set_channel(lr, hr, n_channels=self.args.n_colors)
        tensors = common.np2Tensor(lr, hr, rgb_range=self.args.rgb_range)
        return tensors[0], tensors[1], fname

    def __len__(self):
        return len(self.hr_list) * (self.repeat_factor if self.is_train else 1)

    # ---------------------------------------------------------------------
    def _read_pair(self, index):
        """Load HR-LR pair from image or binary cache."""
        idx = index % len(self.hr_list) if self.is_train else index
        hr_path, lr_path = self.images_hr[idx], self.images_lr[self.current_scale_index][idx]
        fname, _ = os.path.splitext(os.path.basename(hr_path))

        if self.args.ext == "img" or self.benchmark:
            hr, lr = imageio.imread(hr_path), imageio.imread(lr_path)
        else:
            with open(hr_path, "rb") as f: hr = pickle.load(f)
            with open(lr_path, "rb") as f: lr = pickle.load(f)

        return lr, hr, fname

    # ---------------------------------------------------------------------
    def _prepare_patch(self, lr, hr):
        """Extract training patches or align evaluation pairs."""
        scale = self.scale_factors[self.current_scale_index]

        if self.is_train:
            lr, hr = common.get_patch(
                lr, hr,
                patch_size=self.args.patch_size,
                scale=scale,
                multi=(len(self.scale_factors) > 1),
                input_large=self.large_input,
            )
            lr, hr = common.augment(lr, hr)
        else:
            ih, iw = lr.shape[:2]
            hr = hr[0 : ih * scale, 0 : iw * scale]

        return lr, hr

    def set_scale(self, idx_scale: int):
        """Update the active scale factor for multi-scale training."""
        if self.large_input:
            self.current_scale_index = random.randint(0, len(self.scale_factors) - 1)
        else:
            self.current_scale_index = idx_scale



