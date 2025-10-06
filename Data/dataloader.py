import torch
import random
import queue
import sys
import threading
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import _DataLoaderIter
from torch.utils.data._utils import (
    collate, signal_handling, MP_STATUS_CHECK_INTERVAL
)
from torch.utils.data._utils.worker import ManagerWatchdog, ExceptionWrapper


def _ms_worker_loop(dataset, task_queue, result_queue, stop_event,
                    collate_fn, scales, seed, init_fn, worker_id):

    try:
        collate._use_shared_memory = True
        signal_handling._set_worker_signal_handlers()

        # Ensure deterministic behavior per worker
        torch.set_num_threads(1)
        random.seed(seed)
        torch.manual_seed(seed)

        if init_fn is not None:
            init_fn(worker_id)

        result_queue.cancel_join_thread()
        watchdog = ManagerWatchdog()

        while watchdog.is_alive():
            try:
                task = task_queue.get(timeout=MP_STATUS_CHECK_INTERVAL)
            except queue.Empty:
                continue

            # Graceful exit condition
            if task is None or stop_event.is_set():
                if stop_event.is_set():
                    return
                continue

            batch_idx, sample_indices = task

            try:
                # Random scale assignment during training
                scale_index = 0
                if getattr(dataset, "train", False) and len(scales) > 1:
                    scale_index = random.randint(0, len(scales) - 1)
                    dataset.set_scale(scale_index)

                batch_samples = [dataset[i] for i in sample_indices]
                collated = collate_fn(batch_samples)
                collated.append(scale_index)

                result_queue.put((batch_idx, collated))

            except Exception:
                wrapped = ExceptionWrapper(sys.exc_info())
                result_queue.put((batch_idx, wrapped))
            finally:
                del batch_samples, collated

    except KeyboardInterrupt:
        print(f"[Worker {worker_id}] Interrupted — shutting down gracefully.")
    except Exception as e:
        wrapped = ExceptionWrapper(sys.exc_info())
        result_queue.put((None, wrapped))


class _MSLoaderIterator(_DataLoaderIter):

    def __init__(self, loader):
        super().__init__(loader)
        self.dataset = loader.dataset
        self.scales = loader.scales
        self.collate_fn = loader.collate_fn
        self.batch_sampler = loader.batch_sampler
        self.num_workers = loader.num_workers
        self.pin_memory = loader.pin_memory and torch.cuda.is_available()
        self.timeout = loader.timeout
        self.worker_init_fn = loader.worker_init_fn
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Internal state variables
        self.shutdown_flag = False
        self.index_iter = iter(self.batch_sampler)
        self.result_queue = mp.Queue()
        self.stop_event = mp.Event()

        self._launch_workers()
        self._start_prefetching()

    # ----------------------------------------------------------------------
    def _launch_workers(self):
        """Initialize and start worker processes."""
        self.workers = []
        self.task_queues = []
        base_seed = torch.randint(0, 2**30, (1,)).item()

        for wid in range(self.num_workers):
            q = mp.Queue()
            q.cancel_join_thread()

            worker = mp.Process(
                target=_ms_worker_loop,
                args=(
                    self.dataset,
                    q,
                    self.result_queue,
                    self.stop_event,
                    self.collate_fn,
                    self.scales,
                    base_seed + wid,
                    self.worker_init_fn,
                    wid,
                ),
            )
            worker.daemon = True
            worker.start()
            self.workers.append(worker)
            self.task_queues.append(q)

        signal_handling._set_worker_pids(
            id(self), tuple(w.pid for w in self.workers)
        )
        signal_handling._set_SIGCHLD_handler()

    # ----------------------------------------------------------------------
    def _start_prefetching(self):
        """Enqueue initial tasks for prefetching."""
        for _ in range(2 * self.num_workers):
            self._enqueue_next_batch()

        if self.pin_memory:
            self.data_queue = queue.Queue()
            self.pin_thread = threading.Thread(
                target=self._pin_memory_thread,
                args=(),
                daemon=True
            )
            self.pin_thread.start()
        else:
            self.data_queue = self.result_queue

    # ----------------------------------------------------------------------
    def _enqueue_next_batch(self):
        """Send a new batch to an available worker queue."""
        try:
            next_batch = next(self.index_iter)
            task = (id(next_batch), list(next_batch))
            q = self.task_queues[
                (len(self.task_queues) + hash(task[0])) % len(self.task_queues)
            ]
            q.put(task)
        except StopIteration:
            pass

    # ----------------------------------------------------------------------
    def _pin_memory_thread(self):
        """Background thread to transfer tensors to pinned memory."""
        from torch.utils.data._utils.pin_memory import _pin_memory_loop
        _pin_memory_loop(
            in_queue=self.result_queue,
            out_queue=self.data_queue,
            device_id=torch.cuda.current_device(),
            done_event=self.stop_event
        )

    # ----------------------------------------------------------------------
    def __next__(self):
        """Retrieve the next batch, handling out-of-order results."""
        if self.shutdown_flag:
            raise StopIteration

        try:
            batch = self.data_queue.get(timeout=self.timeout)
        except queue.Empty:
            self._shutdown()
            raise StopIteration

        if isinstance(batch[1], ExceptionWrapper):
            batch[1].reraise()

        self._enqueue_next_batch()
        return batch[1]

    # ----------------------------------------------------------------------
    def _shutdown(self):
        """ shut down all worker processes."""
        if self.shutdown_flag:
            return
        self.shutdown_flag = True
        self.stop_event.set()

        for q in self.task_queues:
            q.put(None)
        for w in self.workers:
            w.join(timeout=1.0)


class MultiScaleDataLoader(DataLoader):
    """
    Parameters
    ----------
    cfg : argparse.Namespace or similar
        Configuration object with scale and threading parameters.
    *args, **kwargs : Any
        Standard DataLoader arguments.
    """

    def __init__(self, cfg, *args, **kwargs):
        super().__init__(*args, **kwargs, num_workers=cfg.n_threads)
        self.scales = cfg.scale if hasattr(cfg, "scale") else [1]

    def __iter__(self):
        return _MSLoaderIterator(self)


