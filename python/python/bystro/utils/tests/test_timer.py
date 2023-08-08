import time

from bystro.utils.timer import Timer


def test_Timer():
    sleep_time = 0.1
    with Timer() as timer:
        time.sleep(sleep_time)
    relative_error = (timer.elapsed_time - sleep_time) / sleep_time
    assert abs(relative_error) < 0.1
