import time

from bystro.utils.timer import Timer


def test_Timer():
    sleep_time = 1.0  # Increased sleep time to 1 second
    iterations = 5  # Run 5 iterations and average the results
    total_elapsed = 0.0

    for _ in range(iterations):
        with Timer() as timer:
            time.sleep(sleep_time)
        total_elapsed += timer.elapsed_time

    average_elapsed = total_elapsed / iterations
    relative_error = (average_elapsed - sleep_time) / sleep_time

    # Log diagnostic info
    print(
        (
            f"Expected Sleep Time: {sleep_time}, "
            f"Average Elapsed Time: {average_elapsed}, "
            f"Relative Error: {relative_error}"
        )
    )

    # Use a higher tolerance to account for CI environment variations
    assert abs(relative_error) < 0.3
