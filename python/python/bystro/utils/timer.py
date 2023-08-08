import time


class Timer:
    def __enter__(self):
        """Start a new timer as a context manager"""
        self.start_time = time.time()
        return self

    def __exit__(self, *exc_info):
        """Stop the context manager timer"""
        self.elapsed_time = time.time() - self.start_time
