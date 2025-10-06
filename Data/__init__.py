import os
import torch
import numpy as np
import random
import importlib
from datetime import datetime
from torch.utils.data import DataLoader, ConcatDataset, get_worker_info


class MultiScaleConcatDataset(ConcatDataset):
    """
    Combines multiple datasets into one unified dataset.
    Each sub-dataset may have its own scale and augmentation.
    """

    def __init__(self, datasets):
        super().__init__(datasets)
        self.train = getattr(datasets[0], "train", True)
        self.dataset_names = [getattr(d, "name", "Unnamed") for d in datasets]

    def __repr__(self):
        desc = f"MultiScaleConcatDataset({len(self.datasets)} datasets: "
        desc += ", ".join(self.dataset_names) + ")"
        return desc

    def set_scale(self, idx_scale):
        """Propagate scale changes to all sub-datasets."""
        for d in self.datasets:
            if hasattr(d, "set_scale"):
                d.set_scale(idx_scale)


class DataManager:
    """
    Handles all dataset and DataLoader creation logic for both training and testing.
    Supports:
        - Multi-dataset training
        - Dynamic scale switching
        - Automatic threading adjustments
        - Safe multiprocessing initialization
    """

    def __init__(self, args):
        self.args = args
        self.loader_train = None
        self.loader_test = []
        self._init_seed(args.seed)
        self._print_header()

        if not args.test_only:
            self.loader_train = self._build_train_loader(args)

        self.loader_test = self._build_test_loaders(args)
        self._print_summary()

    def _build_train_loader(self, args):
        """Constructs unified training DataLoader."""
        print("Building training datasets...")

        train_datasets = []
        for dataset_name in args.data_train:
            try:
                module = importlib.import_module(f"data.{dataset_name.lower()}")
                dataset_class = getattr(module, dataset_name)
                ds = dataset_class(args, name=dataset_name)
                train_datasets.append(ds)
                print(f"Loaded {dataset_name} ({len(ds)} samples)")
            except Exception as e:
                print(f"  ⚠️ Failed to load {dataset_name}: {e}")

        # Combine all training datasets into one
        combined_dataset = MultiScaleConcatDataset(train_datasets)
        loader = DataLoader(
            dataset=combined_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=self._auto_workers(),
            pin_memory=not args.cpu,
            drop_last=True
        )

        print(f"Combined training dataset ready ({len(combined_dataset)} total samples)")
        return loader

    def _build_test_loaders(self, args):
        """Constructs test DataLoaders for all datasets."""
        print("Preparing test datasets...")
        test_loaders = []

        for dataset_name in args.data_test:
            try:
                module = importlib.import_module(f"data.{dataset_name.lower()}")
                dataset_class = getattr(module, dataset_name)
                test_ds = dataset_class(args, train=False, name=dataset_name)
                loader = DataLoader(
                    dataset=test_ds,
                    batch_size=1,
                    shuffle=False,
                    num_workers=self._auto_workers(),
                    pin_memory=not args.cpu
                )
                test_loaders.append(loader)
                print(f"Test dataset '{dataset_name}' with {len(test_ds)} images")
            except Exception as e:
                print(f"Could not initialize {dataset_name}: {e}")

        return test_loaders

    def _auto_workers(self):
        """Automatically decide number of workers based on available CPUs."""
        import multiprocessing
        max_workers = multiprocessing.cpu_count()
        if hasattr(self.args, "n_threads") and self.args.n_threads > 0:
            return min(self.args.n_threads, max_workers)
        else:
            return max(2, max_workers // 2)

    def _init_seed(self, seed):
        """Initialize reproducible random seed."""
        import torch, numpy as np
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _print_summary(self):
        if self.loader_train:
            train_size = len(self.loader_train.dataset)
            print(f"  🏋️ Training samples: {train_size}")
            print(f"  Batch size: {self.args.batch_size}")
            print(f"  Num workers: {self._auto_workers()}")
        print(f"  🧪 Test datasets: {len(self.loader_test)}")
        for i, loader in enumerate(self.loader_test):
            print(f"    {i+1}. {loader.dataset.name} → {len(loader.dataset)} images")

    def set_scale(self, idx_scale):
        """Manually set scale for multi-scale datasets."""
        if self.loader_train and hasattr(self.loader_train.dataset, "set_scale"):
            self.loader_train.dataset.set_scale(idx_scale)
        for loader in self.loader_test:
            if hasattr(loader.dataset, "set_scale"):
                loader.dataset.set_scale(idx_scale)

  
    @staticmethod
    def worker_init_fn(worker_id):
        """Initialize workers deterministically for reproducibility."""
        info = get_worker_info()
        dataset = info.dataset
        base_seed = torch.initial_seed() % 2**32
        random.seed(base_seed + worker_id)
        np.random.seed(base_seed + worker_id)
        if hasattr(dataset, "set_scale"):
            dataset.set_scale(0)


