"""Provide a timer as a contextmanager."""

import time


class Timer:
    def __enter__(self):
        """Start a new timer as a context manager."""
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *exc_info):
        """Stop the timer."""
        self.elapsed_time = time.perf_counter() - self.start_time
