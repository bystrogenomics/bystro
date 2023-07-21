"""Convenience methods for safe asserts."""

from typing import Any


def assert_true(
    description: str,
    condition: Any,  # noqa: ANN401  Any is actually appropriate here
    comment: str = "",
) -> None:
    """Check that condition holds, raising AssertionError if not."""
    if not condition:
        msg = f"Expected {description}."
        if comment:
            msg += "  " + comment
        raise AssertionError(msg)


def assert_equals(
    expected_description: str,
    expected_value: Any,  # noqa: ANN401
    actual_description: str,
    actual_value: Any,  # noqa: ANN401
    comment: str = "",
) -> None:
    """Check that expected_value equals actual_value, raising AssertionError if not."""
    comparison = expected_value == actual_value
    success = all(comparison) if hasattr(comparison, "__len__") else comparison
    if success:
        return
    msg = (
        f"Expected {expected_description} ({expected_value}) "
        f"to equal {actual_description} ({actual_value})."
    )
    if comment:
        msg += "  " + comment
    raise AssertionError(msg)
