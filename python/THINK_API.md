# Bystro Think Python API

The Think SDK submits durable agent workloads, streams visible output and
structured progress, handles human-input pauses, uploads large files in chunks,
reuses Bystro datasets and conversations as context, and downloads protected
results.

## Install

For agent workloads only, install the lightweight client. It supports CPython
3.11 and newer, including 3.13, and does not install the genomics/scientific
stack:

```sh
python --version
python -m pip install "bystro-think==2.1.3"
```

For both Think and Bystro's local genomics tools, install the full distribution
on CPython 3.11 or 3.12:

```sh
python --version
python -m pip install "bystro==2.1.3"
```

Choose one distribution per environment. Both intentionally provide the same
`bystro.think` and `bystro.api.auth` imports, so they should not be co-installed.

Production uses publicly trusted HTTPS certificates. Customers do not need a
custom CA bundle or TLS override; those are only for local development servers
using a private certificate authority.

## Authenticate once, then use the cached login

Use `getpass` for the one-time interactive login so secrets do not appear in
source code, notebook output, shell history, or environment listings:

```python
from getpass import getpass

from bystro.api import auth


email = input("Bystro email: ").strip()
site_access_code = getpass("Site access code (leave blank if not required): ")
auth.login(
    email,
    getpass("Bystro password: "),
    site_access_code=site_access_code or None,
)
```

The login JWT is stored in `~/.bystro/bystro_authentication_token.json`; the
directory is mode `0700` and the atomically replaced file is mode `0600`. The
site-access code is used only for that login session and is not cached.

Normal scripts then use the cached login without handling a password:

```python
from bystro.think import ThinkClient

client = ThinkClient.from_cached_login()
```

New accounts must accept the current legal assertions once. Use
`LegalConsent.accepted(name)` with `auth.signup(...)`, or complete signup in the
dashboard before running the login snippet above.

## Canonical interactive workflow

This is the recommended customer experience. It prints lifecycle changes,
backend-owned phases when the service emits them, visible answer chunks, and an
elapsed heartbeat if no server frame arrives for 30 seconds. `interact()`
prompts for any number of clarification or plan-review pauses.

```python
from bystro.think import (
    BillingTopUpApproval,
    BillingTopUpRequest,
    NeedsInput,
    RunResult,
    ThinkClient,
    show_progress,
)


def approve_top_up(
    request: BillingTopUpRequest,
) -> BillingTopUpApproval | None:
    amount = request.minimum_top_up_cents
    dollars = f"{amount // 100}.{amount % 100:02d}"
    answer = input(f"This message needs a ${dollars} top-up. Approve? [y/N] ")
    return request.approve(amount) if answer.strip().lower() == "y" else None


with ThinkClient.from_cached_login(
    on_event=show_progress,
    on_billing_required=approve_top_up,
) as client:
    run = client.submit_with_progress(
        "Research the latest CAR-T therapies and cite primary sources."
    )
    outcome = run.interact(timeout=3600)

    if isinstance(outcome, NeedsInput):
        # interact() handles clarification and plan-review pauses itself.
        # Admission top-ups are handled above. A returned NeedsInput is a
        # mid-run billing pause; resolve its durable operation in the dashboard
        # and call run.refresh().
        print(outcome)
    else:
        assert isinstance(outcome, RunResult)
        print("\nFinal Markdown is also available as outcome.output")
```

`submit_with_progress()` installs a progress renderer automatically. Supplying
`show_progress` on the client also includes connection and reconnect events.
Do not pass the same callback again to `run.wait(on_event=...)`.

The billing callback is invoked only after the server prices and rejects the
specific message. It must return `request.approve(...)` or `None`; the SDK never
infers consent, never accepts less than `minimum_top_up_cents`, and makes at most
one top-up attempt for that submission. The approved amount raises the fixed
monthly extra-usage cap; the actual usage charge remains part of the retried
conversation reservation. If Stripe needs a payment method or billing-address
update, `ThinkBillingRequiredError.action_url` contains the hosted URL and the
blocked message is not retried. Returning `None` declines the proposal and
raises that same typed error without changing the cap.

Think uses authenticated Socket.IO transport. It normally upgrades to WebSocket
and retains HTTP polling as a compatibility fallback. Output frames are
cumulative short snapshots, not necessarily one event per tokenizer token; the
SDK turns them into exact append/replace/retract updates and never exposes
internal reasoning text.

To require native WebSocket and fail rather than fall back to polling:

```python
with ThinkClient.from_cached_login(
    on_event=show_progress,
    transports=("websocket",),
) as client:
    result = client.submit_with_progress("Draw a duck.").interact(timeout=3600)
```

## Non-interactive input callbacks

Applications can answer pauses without calling `input()`. Manual `wait()` and
`respond()` remain available when the application needs complete control.

```python
from bystro.think import NeedsInput, ThinkClient


def answer_clarification(request: NeedsInput) -> str:
    print("Clarification:", request.prompt)
    return "Cover all disease areas and the last 24 months."


def review_plan(request: NeedsInput) -> str:
    print("Proposed plan:", request.prompt)
    return "accept"


with ThinkClient.from_cached_login() as client:
    run = client.submit_with_progress("Research recent CAR-T therapies.")
    result = run.interact(
        timeout=3600,
        on_clarification=answer_clarification,
        on_plan_review=review_plan,
    )
```

For manual control:

```python
from bystro.think import InputKind, NeedsInput


outcome = run.wait(timeout=3600)
if isinstance(outcome, NeedsInput):
    if outcome.kind is InputKind.PLAN_REVIEW:
        run.respond("accept")
    elif outcome.kind is InputKind.CLARIFICATION:
        run.respond("Use case_control as the phenotype column")
```

The first live pause can precede its durable checkpoint commit. `respond()`
waits for the checkpoint replay before uploading attachments or dispatching the
answer, and ignores stale replayed checkpoints after reconnect.

## Choose a mode

The default mode is `base`. Pass `RunOptions` per submitted conversation:

```python
from bystro.think import RunOptions


run = client.submit_with_progress(
    "Research the latest CAR-T therapies and cite primary sources.",
    options=RunOptions(mode="plus2"),
)
```

| Value | Dashboard name | Intended use |
| --- | --- | --- |
| `base` | Base | Faster, token-efficient work with lighter research. |
| `plus` | Plus v1 | Verified analysis with deep research. |
| `plus2` | Plus v2 | Stronger experimental research workflow. |
| `phd` | PhD | Deepest reasoning for demanding analyses. |

Other controls are typed fields on `RunOptions`: `advanced_planning`,
`auto_compact`, `fast`, `verify`, `verify_sources`, and
`zero_data_retention`. Availability and billing follow the authenticated
account and deployment configuration.

## Submit files with the question

A path passed in `files` is uploaded to the authenticated user's personal
artifacts and attached to the same message. Large inputs use resumable bounded
10 MiB chunks, SHA-256 checksums, idempotent retries, and asynchronous
finalization polling.

```python
from bystro.think import ThinkClient, UploadProgress


def upload_progress(progress: UploadProgress) -> None:
    print(
        f"[upload:{progress.phase.value}] {progress.fraction:.0%}",
        flush=True,
    )


with ThinkClient.from_cached_login() as client:
    run = client.submit_with_progress(
        "Analyze the cohort using the attached phenotype table.",
        files=["cohort.vcf.gz", "phenotypes.tsv"],
        on_upload_progress=upload_progress,
    )
    result = run.interact(timeout=3600)
```

A single path can be passed directly:

```python
run = client.submit_with_progress(
    "Summarize this study protocol.",
    files="protocol.pdf",
)
```

Create a reusable artifact before submission with `upload_artifact()` (or its
short alias `upload()`):

```python
artifact = client.upload_artifact(
    "cohort.vcf.gz",
    artifact_path="study/cohort.vcf.gz",
    on_progress=upload_progress,
)
run = client.submit_with_progress("Run QC on this cohort.", files=[artifact])
```

Artifact paths are relative, have at most 64 components, and must end with the
local file's exact name. Invalid paths fail locally before upload.

## Compose genetic, conversation, and artifact context

The context helpers accept either a string or an immutable
`MessageWithContext`, so they compose without constructing XML manually. The
SDK serializes an escaped XML preview with `message.to_xml()`, while the live
request keeps ownership-bearing references in structured metadata.

```python
from bystro.think import (
    add_artifact_context,
    add_genetic_context,
    add_previous_conversation_context,
)


message = "Re-evaluate the strongest phenotype associations."
message = add_genetic_context(
    "annotation-job-id",
    message,
    name="Case cohort",
    assembly="hg38",
)
message = add_previous_conversation_context(
    "prior-thread-id",
    message,
    name="Earlier analysis",
)
message = add_artifact_context(artifact, message)

run = client.submit_with_progress(message)
```

Reusable higher-order transforms are available:

```python
from bystro.think import (
    artifact_context,
    compose_context,
    genetic_context,
    previous_conversation_context,
)


study_context = compose_context(
    genetic_context("annotation-job-id", assembly="hg38"),
    previous_conversation_context("prior-thread-id"),
    artifact_context("existing-artifact-id"),
)

run = client.submit_with_progress(
    study_context("Compare the strongest signals.")
)
```

Think resolves every dataset, conversation, and artifact under the
authenticated user's ownership. User-authored XML is never an authorization
boundary.

## Structured progress and custom presentation

`ThinkEvent.progress` contains the server's current phase snapshot. Typical
phase kinds are `search`, `verify`, `compute`, `query`, and `think`.
`ThinkEvent.stream_update` contains safe visible-output deltas.

```python
from bystro.think import EventKind, ThinkEvent


def on_event(event: ThinkEvent) -> None:
    if event.progress is not None:
        phase = event.progress.active_phase
        if phase is not None:
            print(phase.kind, phase.label, phase.completed, phase.total)
        return

    update = event.stream_update
    if event.kind is EventKind.STREAM and update is not None:
        if update.operation == "append":
            print(update.delta, end="", flush=True)
        elif update.operation == "replace":
            print("\n[corrected output]\n", update.delta)
        elif update.operation == "retract":
            print(f"\n[removed message {update.message_id}]")
```

`ProgressRenderer(heartbeat_interval=30)` provides the canonical terminal
presentation. Generic `Thinking...` and `Processing...` states print at most
once per turn; meaningful phase/count changes print immediately; and, during
transport silence, the renderer repeats a structured phase only while the
backend reports it active or pending. Otherwise it emits
`Still working... (… elapsed)`. Heartbeat workers stop on input, completion,
failure, cancellation, or local detach.

## Results, conversations, and downloads

Generated files are available directly on a successful `RunResult`. The
listing is authenticated and loaded once, on first access, so access it while
the client context is open:

```python
from pathlib import Path

from bystro.think import RunResult, ThinkClient


with ThinkClient.from_cached_login() as client:
    run = client.submit_with_progress("Draw and save a cartoon duck.")
    result = run.interact(timeout=3600)
    if not isinstance(result, RunResult):
        raise RuntimeError("The run paused for billing")

    print("Mode:", result.mode)
    print("Started:", result.execution_started_at)
    print("Completed:", result.execution_completed_at)
    print("Execution seconds:", result.execution_duration_seconds)

    for output_file in result.files:  # result.artifacts is the same tuple
        print(output_file.path, output_file.size)

    if result.files:
        first = run.download_file(
            result.files[0],
            Path("downloads") / result.files[0].path,
        )
        print("Downloaded:", first)

    archive = run.download_all(Path("downloads") / f"{run.id}.tar")
    print("Archive:", archive)
```

`download_file()` and `download_all()` stream to a temporary file and publish
the destination only after the authenticated download completes. Existing
targets are never replaced unless `overwrite=True` is explicit.

`result.options` contains the complete typed `RunOptions` used for the turn;
`result.mode` is its convenient mode alias. Execution timing comes from the
durable final-message metadata and is `None` only when an older transcript does
not contain that field. `result.files` is the authenticated output-file
manifest and remains lazily loaded so text-only callers do not pay for another
request.

List and resume past conversations:

```python
conversations = client.list_conversations(search="CAR-T", limit=20)
for conversation in conversations:
    print(conversation.id, conversation.name, conversation.created_at)

previous = client.resume(conversations[0].id)
previous_outcome = previous.wait(timeout=60)
print(previous.messages)
print(previous.output_files())
```

Omit `limit` to traverse all cursor pages. `run.messages` excludes internal
reasoning and progress-card messages; `run.history` is bounded SDK event
history. `resume()` starts transcript replay asynchronously; call `wait()` before
reading a completed or paused conversation. For work that is still active,
iterate `previous.events()` to observe it through its next pause or completion.
A resumed run restores its submitted mode and other `RunOptions`, so
`run.follow_up(...)` continues with the original settings.

In Jupyter, `RunResult` and `NeedsInput` implement `_repr_markdown_()`, so
placing either object at the end of a cell renders its Markdown naturally.

## Cancellation, detach, and reconnect

Cancellation is distinct from closing a local client:

```python
from bystro.think import RunCancelledError


run.cancel(timeout=60)  # waits for durable server cleanup to be released
try:
    run.wait()
except RunCancelledError:
    print("Cancelled")
```

`cancel()` sends the active task ID when available, ignores delayed lifecycle
events from older tasks, and reissues an interrupted stop after reconnect until
the server emits its durable release event.

Use `run.detach()` (or close the `ThinkClient`) to disconnect locally while the
server keeps working. Reattach from another process later:

```python
run_id = run.id
run.detach()

with ThinkClient.from_cached_login() as client:
    resumed = client.resume(run_id)
    outcome = resumed.wait(timeout=3600)
```

A `ThinkClient` owns one foreground conversation at a time. Use separate
clients for concurrently controlled conversations.

## Caller-controlled idempotency

The SDK automatically reuses one message ID for its own transport retries. For
recovery after a caller process exits before receiving the server's
acknowledgement, persist an idempotency key before submission and reuse it:

```python
from uuid import uuid4


request_id = str(uuid4())  # persist beside the customer job before submitting
run = client.submit_with_progress(
    "Research recent CAR-T approvals.",
    idempotency_key=request_id,
)
```

If the caller cannot tell whether that message was accepted, submitting it
again with the same key resolves to the original durable admission instead of
starting a second expensive job. `respond()` and `follow_up()` accept the same
argument. A key identifies one logical message: never reuse it with different
content and expect the new content to run.

## Async applications

The submission API is synchronous; the live event iterator and terminal wait
also have event-loop-friendly async forms:

```python
import asyncio

from bystro.think import ThinkClient


async def main() -> None:
    with ThinkClient.from_cached_login() as client:
        run = client.submit("Research recent CAR-T approvals.")
        async for event in run.aevents(timeout=3600):
            print(event.kind.value)
        result = await run.await_result(timeout=30)
        print(result)


asyncio.run(main())
```

`aevents()` never blocks the event loop while waiting for Socket.IO events;
durable refreshes run outside the loop.

## Cloudflare configuration

No Cloudflare change is needed when the installed SDK connects, uploads, and
downloads successfully. If browser-only challenges intercept Python traffic,
create a **zone-level custom rule** with action **Skip** and match only the API
hosts/routes used by the SDK. Select only:

- All Super Bot Fight Mode rules
- Browser Integrity Check
- Security Level

Keep **Log matching requests** enabled. Do not select all remaining custom
rules, rate limiting rules, or managed WAF rules unless a specific logged false
positive proves one of those components is responsible. Cloudflare documents
that Skip can target these products independently, leaving other security
layers active: [Skip action](https://developers.cloudflare.com/waf/custom-rules/skip/)
and [available skip options](https://developers.cloudflare.com/waf/custom-rules/skip/options/).

For `ai.bystro.cloud`, the complete SDK transport surface is:

```text
/auth/cookie
/set-session-cookie
/ws/socket.io
/project/threads
/user/billing/spend-cap
/user/files/*
/api/user-output/*
```

The approval callback specifically requires a matching `PUT` rule for the
exact `/user/billing/spend-cap` path.

For `bystro.cloud`, one-time programmatic login uses:

```text
/api/site-gate/authenticate
/api/user/auth/local
```

If customers also need to discover existing genetic-analysis jobs with
`bystro.api.annotation.get_jobs`, include the narrow `/api/jobs*` path prefix
on `bystro.cloud`. This route is optional when the caller already knows the
genetic job ID. It remains protected by Bystro authentication and authorization;
the Cloudflare rule skips only the selected browser-oriented checks.

Keep the route expression narrow rather than bypassing by user agent or a
shared customer token. Application authentication and ownership checks still
run at the origin, while Cloudflare DDoS protection, managed WAF, and rate
limits remain available.

An easy verification is to install the wheel in a clean environment, unset
`REQUESTS_CA_BUNDLE` and `SSL_CERT_FILE`, require
`transports=("websocket",)`, submit a small run, list conversations, upload a
file larger than 10 MiB, and download one result plus the tar archive. A
Cloudflare HTML challenge or `cf-ray` 403 indicates the route rule still does
not match; a typed JSON/application error means the request reached Bystro.

## Errors

Transport, HTTP, authentication, billing, admission, cancellation, timeout, and
protocol failures have typed exceptions under `bystro.think`. In particular:

- `ThinkAuthenticationError`: cached dashboard login is missing or expired.
- `ThinkBillingRequiredError`: admission requires billing action. Its `request`
  holds a typed top-up proposal when one can be approved programmatically, and
  `action_url` identifies any required Stripe-hosted setup.
- `RunRejectedError`: submission was rejected before dispatch.
- `RunTimeoutError`: a local wait deadline elapsed; the durable run may continue.
- `RunCancelledError`: server-side cancellation completed.
- `RunFailedError`: the accepted workload failed during server execution.
- `RunProtocolError`: the server returned contradictory or incomplete state.

Closing a client never implies cancellation.
