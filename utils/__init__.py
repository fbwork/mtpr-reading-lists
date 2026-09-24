from utils.cleaning import *
import progressbar
from time import sleep
import functools
import os
from multiprocessing import Pool, Value, Manager
import logging
import numpy as np  # Added for numpy iterable check


def gram_len(gram: str):
    return gram.count(' ') + 1


class ProgressBar:
    def __init__(self, status, max_value, chunk_size=None):
        self.status = status or 'loading...'
        self.chunk_size = chunk_size
        self.max_value = max_value
        self.progress_level = 0
        self.bar = self.get_progressbar()
        self.is_finish = False

    @property
    def get_widget(self):
        return [self.status, progressbar.Bar('*'), ' (', progressbar.ETA(), ')']

    def get_progressbar(self):
        return progressbar.ProgressBar(maxval=self.max_value, widgets=self.get_widget)

    def start(self):
        # Reduced sleep for faster start - adjust if needed
        sleep(0.1)
        self.bar.start()
        sleep(0.1)

    def update(self, chunk_size=None):
        if self.max_value <= 0:
            # Skip update if max_value is invalid
            return

        chunk = chunk_size or self.chunk_size or 1
        if chunk <= 0:
            raise ValueError('Chunk size must be positive.')

        self.progress_level += chunk
        self.progress_level = min(self.progress_level, self.max_value)  # Prevent overflow

        if self.progress_level >= self.max_value:
            self.finish()
        else:
            self.bar.update(self.progress_level)

    def finish(self):
        if not self.is_finish:
            self.is_finish = True
            self.bar.finish()
            sleep(0.5)  # Shorter sleep for finish


class TaskAssistant:
    """
    A mini framework in the form of function designed for introducing a processing task along with its inputs and
    callback function. This function can operate in both single- or multiprocessing modes.
    """

    def __init__(self, task: callable, kw_data: dict, status: str, callback_func: callable,
                 callback_kwargs: dict = None,
                 task_kwargs: dict = None, chunk_size=2000, mp_core_frac=1):
        self.is_mp_enabled = mp_core_frac
        self.task = functools.partial(task, **(task_kwargs or {}))
        self.kw_data = kw_data
        self.chunk_size = chunk_size

        # Separate iterable and scalar arguments
        self.iterable_keys = [k for k in kw_data if self._is_iterable(kw_data[k])]
        self.scalar_kwargs = {k: v for k, v in kw_data.items() if k not in self.iterable_keys}
        self.iterable_data = [kw_data[k] for k in self.iterable_keys]

        # Calculate num_items based on first iterable (or 1 if none)
        num_items = len(self.iterable_data[0]) if self.iterable_data else 1
        self.bar = ProgressBar(status, num_items, chunk_size)

        # Partial callback with bar
        self.callback = functools.partial(callback_func, **(callback_kwargs or {}), bar=self.bar)
        self.logger = logging.getLogger()
        self.mp_core_frac = False if not mp_core_frac else (
            os.cpu_count() - 1 if mp_core_frac == 1 else int(os.cpu_count() * mp_core_frac))

    def _is_iterable(self, obj):
        # Treat strings and scalars as non-iterable; support lists, tuples, numpy arrays
        if isinstance(obj, (str, int, float, np.float64)):
            return False
        if isinstance(obj, np.ndarray):
            return obj.size > 0
        return isinstance(obj, (list, tuple, range)) and len(obj) > 0

    @staticmethod
    def MP_Value(*args, **kwargs):
        return Value(*args, **kwargs)

    @staticmethod
    def create_list():
        return Manager().list()

    def execute(self):
        # Handle case with no iterables (single task)
        if not self.iterable_data:
            kw = self.scalar_kwargs.copy()
            if self.bar.max_value > 0:
                self.bar.start()
            if self.mp_core_frac:
                with Pool(processes=1) as pool:
                    pool.apply_async(self.task, kwds=kw, callback=self.callback)
                    pool.close()
                    pool.join()
            else:
                result = self.task(**kw)
                self.callback(result)
            if self.bar.max_value > 0:
                self.bar.finish()
            return

        all_data = self.iterable_data
        all_kws = self.iterable_keys

        # Apply chunking if enabled and data is large enough
        chunked_data = all_data
        if self.chunk_size != 1 and len(all_data[0]) > self.chunk_size:
            buckets = max(1, round(len(all_data[0]) / self.chunk_size))
            chunked_data = [[ad[i::buckets] for i in range(buckets)] for ad in all_data]
            # Do NOT reset max_value here—keep it as original num_items for per-item progress
            self.bar.progress_level = 0  # Reset progress

        if self.bar.max_value > 0:
            self.bar.start()

        if self.mp_core_frac:
            with Pool(processes=self.mp_core_frac) as pool:
                for data in zip(*chunked_data):
                    kw = dict(zip(all_kws, data))
                    kw.update(self.scalar_kwargs)
                    pool.apply_async(self.task, kwds=kw, callback=self.callback)
                pool.close()
                pool.join()
        else:
            for data in zip(*chunked_data):
                kw = dict(zip(all_kws, data))
                kw.update(self.scalar_kwargs)
                result = self.task(**kw)
                self.callback(result)

        self.bar.finish()


def setup_logger():
    level = logging.INFO
    logger = logging.getLogger("app")
    logger.setLevel(level)
    log_file = 'try.log'
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)-8s - %(filename)-20s:%(lineno)-5d - %(funcName)-22s - %(message)s')
    ch = logging.FileHandler(log_file)
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.info("Setup logger in PID {}".format(os.getpid()))
    return logger