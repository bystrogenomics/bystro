import time


class Timer:
    def __enter__(self):
        """Start a new timer as a context manager"""
        self.start_time = time.time()
        return self

    def __exit__(self, *exc_info):
        """Stop the context manager timer"""
        self.elapsed_time = time.time() - self.start_time


def test_Timer():
    sleep_time = 0.1
    with Timer() as timer:
        time.sleep(sleep_time)
    relative_error = (timer.elapsed_time - sleep_time) / sleep_time
    assert abs(relative_error) < 0.1
