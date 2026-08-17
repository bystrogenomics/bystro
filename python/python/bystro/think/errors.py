"""Typed exceptions raised by the Bystro Think SDK."""

from __future__ import annotations

from typing import TYPE_CHECKING, Mapping

if TYPE_CHECKING:
    from bystro.think.models import BillingTopUpRequest


class ThinkError(RuntimeError):
    """Base class for Think SDK failures."""


class ThinkHTTPError(ThinkError):
    """A Think HTTP endpoint returned a controlled error."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int,
        code: str | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.retryable = retryable


class ThinkAuthenticationError(ThinkHTTPError):
    """The cached dashboard session is missing, invalid, or expired."""


class ThinkBillingRequiredError(ThinkHTTPError):
    """The account needs an active Think plan or additional credits."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int,
        code: str | None = None,
        retryable: bool = False,
        request: BillingTopUpRequest | None = None,
        action_url: str | None = None,
    ) -> None:
        super().__init__(
            message,
            status_code=status_code,
            code=code,
            retryable=retryable,
        )
        self.request = request
        self.action_url = action_url


class ThinkConnectionError(ThinkError):
    """The live Think transport could not connect."""


class RunRejectedError(ThinkError):
    """The server rejected a run or follow-up before dispatch."""

    def __init__(
        self,
        message: str,
        *,
        code: str | None,
        retryable: bool,
        acknowledgement: Mapping[str, object],
    ) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable
        self.acknowledgement = acknowledgement


class RunTimeoutError(ThinkError, TimeoutError):
    """Waiting for a run outcome exceeded the requested timeout."""


class RunCancelledError(ThinkError):
    """The run was durably cancelled rather than merely detached."""


class RunFailedError(ThinkError):
    """The server accepted a run, but its execution failed."""

    def __init__(
        self,
        message: str,
        *,
        run_id: str,
        message_id: str,
    ) -> None:
        terminal_safe_message = " ".join(
            "".join(
                (
                    character
                    if ord(character) >= 32
                    and not 127 <= ord(character) <= 159
                    else " "
                )
                for character in message
            ).split()
        )
        super().__init__(terminal_safe_message or "Think run failed")
        self.run_id = run_id
        self.message_id = message_id


class RunProtocolError(ThinkError):
    """The live server sent an incomplete or contradictory run state."""


class InputResponseError(ThinkError):
    """A response cannot be applied to the run's current pause."""


__all__ = [
    "InputResponseError",
    "RunCancelledError",
    "RunFailedError",
    "RunProtocolError",
    "RunRejectedError",
    "RunTimeoutError",
    "ThinkAuthenticationError",
    "ThinkBillingRequiredError",
    "ThinkConnectionError",
    "ThinkError",
    "ThinkHTTPError",
]
