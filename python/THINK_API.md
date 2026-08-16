# Bystro Think Python API

The Think SDK is a synchronous, typed client for durable Bystro agent
workloads. It intentionally exposes a small surface: authenticate, upload
artifacts, compose context, submit work, observe progress, answer pauses, and
resume by run ID.

## Install

Bystro 2.1.1 supports CPython 3.11 and 3.12. Activate the environment you want
to use, confirm it with `python --version`, and install from PyPI:

```sh
python -m pip install "bystro>=2.1.1,<2.2"
```

Production endpoints use publicly trusted HTTPS certificates, so no custom CA
bundle or TLS override is required. A private CA bundle is only needed for a
local development deployment that uses its own certificate authority.

## Authenticate

`auth.login` uses the same `bystro.cloud` dashboard account as the browser and
caches its JWT in `~/.bystro/bystro_authentication_token.json`. The directory
is mode `0700`, the file is atomically replaced at mode `0600`, and tokens are
never printed.

```python
from bystro.api import auth
from bystro.think import ThinkClient

auth.login("you@example.com", "your-password")
client = ThinkClient.from_cached_login()
```

For short scripts, login and construction can be one call:

```python
client = ThinkClient.login("you@example.com", "your-password")
```

If the deployment keeps Bystro's shared site-access gate enabled, present its
code during the same login. The SDK establishes the gate cookie and performs
dashboard login in one private session; the code is never written to the auth
cache:

```python
import os

client = ThinkClient.login(
    "you@example.com",
    "your-password",
    site_access_code=os.environ["BYSTRO_SITE_ACCESS_CODE"],
)
```

New accounts must explicitly provide the dashboard's signed legal assertions:

```python
from bystro.api.auth import LegalConsent, signup

signup(
    "you@example.com",
    "your-password",
    "Your Name",
    legal_consent=LegalConsent.accepted("Your Name"),
    site_access_code=os.environ["BYSTRO_SITE_ACCESS_CODE"],
)
```

`site_access_code` is the Bystro application gate, not a Cloudflare credential.
Cloudflare must still allow non-browser traffic to the dashboard authentication
and Think API routes while retaining its challenge on browser pages.

The SDK exchanges that dashboard session through Think's existing cookie-auth
admission endpoint. The application credential created by Think remains on the
server; it is not copied into local code or exposed as a second API key.

## Customer quickstart: complete examples

Start with this shared setup. Credentials come from environment variables, and
the `wait_for_result` loop handles any number of durable clarification or plan
review pauses while progress continues to stream. A billing pause must be
resolved in the dashboard before refreshing the run.

```python
import os

from bystro.think import (
    InputKind,
    NeedsInput,
    Run,
    RunResult,
    ThinkClient,
    show_progress,
)


def login() -> ThinkClient:
    return ThinkClient.login(
        os.environ["BYSTRO_EMAIL"],
        os.environ["BYSTRO_PASSWORD"],
        site_access_code=os.environ.get("BYSTRO_SITE_ACCESS_CODE"),
        on_event=show_progress,
    )


def wait_for_result(run: Run) -> RunResult:
    while True:
        outcome = run.wait(timeout=3600)
        if not isinstance(outcome, NeedsInput):
            return outcome

        print(f"\n{outcome.prompt}")
        if outcome.kind is InputKind.BILLING:
            input("Resolve billing in the dashboard, then press Enter: ")
            run.refresh()
            continue

        run.respond(input("> "))
```

Set credentials before running an example:

```sh
export BYSTRO_EMAIL="you@example.com"
export BYSTRO_PASSWORD="your-password"
export BYSTRO_SITE_ACCESS_CODE="your-site-code"  # omit when not required
```

### 1. Submit a question with live progress

`submit_with_progress` installs the concise progress renderer automatically.
Passing `show_progress` at login also reports connection and reconnect events.
The final answer remains available as the typed `RunResult.output` value.

```python
with login() as client:
    run = client.submit_with_progress(
        "Research the latest CAR-T therapies and cite primary sources."
    )
    result = wait_for_result(run)

    print("\n--- Final response ---\n")
    print(result.output)
```

Progress is based on durable server lifecycle and status events; it is not a
token-by-token stream of the final prose.

### 2. Submit a question with files

Paths passed through `files` are uploaded first and then attached to the same
question. Large files automatically use the resumable, chunked upload path.

```python
from bystro.think import UploadProgress


def report_upload(progress: UploadProgress) -> None:
    print(
        f"[upload:{progress.phase.value}] {progress.fraction:.0%}",
        flush=True,
    )


with login() as client:
    run = client.submit_with_progress(
        "Analyze the cohort using the attached phenotype table.",
        files=["cohort.vcf.gz", "phenotypes.tsv"],
        on_upload_progress=report_upload,
    )
    result = wait_for_result(run)
    print(result.output)
```

### 3. Submit with genetic, conversation, and artifact context

The `add_*_context` helpers accept a string or an existing
`MessageWithContext`, so context can be built one layer at a time. Existing
references are resolved under the authenticated user's ownership.

```python
import os

from bystro.think import (
    add_artifact_context,
    add_genetic_context,
    add_previous_conversation_context,
)


with login() as client:
    artifact = client.upload_artifact("study-notes.pdf")

    message = "Re-evaluate the strongest phenotype associations."
    message = add_genetic_context(
        os.environ["BYSTRO_JOB_ID"],
        message,
        name="Case cohort",
        assembly="hg38",
    )
    message = add_previous_conversation_context(
        os.environ["BYSTRO_PRIOR_THREAD_ID"],
        message,
        name="Previous analysis",
    )
    message = add_artifact_context(artifact, message)

    run = client.submit_with_progress(message)
    result = wait_for_result(run)
    print(result.output)
```

For a reusable higher-order context pipeline:

```python
from bystro.think import (
    artifact_context,
    compose_context,
    genetic_context,
    previous_conversation_context,
)

add_study_context = compose_context(
    genetic_context("annotation-job-id", assembly="hg38"),
    previous_conversation_context("prior-thread-id"),
    artifact_context("existing-artifact-id"),
)

message = add_study_context("Compare the strongest signals.")
```

### 4. List conversations, browse results, and download files

`list_conversations` returns the authenticated user's conversations newest
first and transparently follows every cursor page. Pass `search` to filter by
conversation name. Resume a returned ID to browse or download its protected
output files through the same authenticated session.

```python
from pathlib import Path


with login() as client:
    conversations = client.list_conversations(search="CAR-T")
    for conversation in conversations:
        print(conversation.id, conversation.name)

    if not conversations:
        raise RuntimeError("No matching conversations")

    previous_run = client.resume(conversations[0].id)
    files = previous_run.output_files()
    for output_file in files:
        print(output_file.path, output_file.size)

    if files:
        downloaded = previous_run.download_file(
            files[0],
            Path("downloads") / files[0].path,
        )
        print("Downloaded:", downloaded)

    archive = previous_run.download_all(
        Path("downloads") / f"{previous_run.id}.tar"
    )
    print("Archive:", archive)
```

`download_file` and `download_all` stream to a temporary file and only publish
the destination after the authenticated download completes. Existing targets
are not overwritten unless `overwrite=True` is passed explicitly.

## Upload files and submit them with a question

Passing paths to `submit` uploads each file to personal input artifacts first,
then attaches the resulting artifact records to the same user message:

```python
run = client.submit(
    "Find variants associated with the case phenotype.",
    files=["cohort.vcf.gz", "phenotypes.tsv"],
)
```

Uploads use the production resumable protocol: bounded 10 MiB chunks,
per-chunk SHA-256 checksums, idempotent retries with exponential backoff, and
polling for asynchronous server finalization. The chunk size and retry policy
can be configured on `ThinkClient`. An `artifact_path` is relative, has at most
64 components, and must end in the local file's exact name; invalid paths fail
locally before authentication or upload begins.

Use `upload_artifact` when the artifact should be created before the question:

```python
def report_upload(progress):
    print(progress.phase.value, f"{progress.fraction:.0%}")

artifact = client.upload_artifact(
    "cohort.vcf.gz",
    artifact_path="study/cohort.vcf.gz",
    on_progress=report_upload,
)
run = client.submit("Run QC on this cohort", files=[artifact])
```

`upload` is an equivalent shorter alias.

## Compose genetic, conversation, and artifact context

Context helpers accept either a plain string or an immutable
`MessageWithContext`, so calls compose naturally:

```python
from bystro.think import (
    add_artifact_context,
    add_genetic_context,
    add_previous_conversation_context,
)

message = "Compare the strongest signals"
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

run = client.submit(message)
```

Reusable higher-order transforms are also available:

```python
from bystro.think import (
    artifact_context,
    compose_context,
    genetic_context,
    previous_conversation_context,
)

study_context = compose_context(
    genetic_context("annotation-job-id", name="Case cohort", assembly="hg38"),
    previous_conversation_context("prior-thread-id"),
    artifact_context("existing-artifact-id"),
)

run = client.submit(study_context("Re-evaluate the phenotype association"))
```

`message.to_xml()` returns a safely escaped preview of the semantic context.
On the wire, references remain structured metadata. Think resolves artifacts
against the authenticated user before creating canonical input-file context;
dataset and conversation retrieval tools independently enforce ownership
before returning referenced data. Raw user-authored XML is never treated as an
ownership boundary.

## Handle `needs_input`

`wait()` returns exactly one of two values: `RunResult` or `NeedsInput`.
Clarifications and plan review are durable checkpoint states, not transient
socket prompts.

```python
from bystro.think import InputKind, NeedsInput

outcome = run.wait(timeout=3600)
if isinstance(outcome, NeedsInput):
    if outcome.kind is InputKind.PLAN_REVIEW:
        run.respond("accept")
    else:
        run.respond("Use case_control as the phenotype column")
    outcome = run.wait(timeout=3600)
```

The first live pause notification can arrive just before its checkpoint is
committed, so `checkpoint_id` may initially be `None`. `run.respond()` requests
the durable replay automatically and will not upload attachments or dispatch
the response until the checkpoint is present. Once answered, replays of that
same or an older checkpoint are ignored, preventing reconnect races from
reopening a stale question. If synchronization fails, the pause remains intact
and `RunProtocolError` asks you to call `run.refresh()` and retry. A billing
pause is also represented as `NeedsInput`, but must be resolved through the
billing action in the dashboard; then call `run.refresh()`.

## Progress and reconnects

`submit_with_progress` uses `show_progress`; use `submit` for silent workloads:

```python
from bystro.think import ThinkClient, show_progress

with ThinkClient.login(
    "you@example.com", "your-password", on_event=show_progress
) as client:
    run = client.submit_with_progress("Perform a GWAS", files=["cohort.vcf.gz"])
    result = run.wait()
```

## Browse and download generated results

```python
conversations = client.list_conversations(search="CAR-T")
for conversation in conversations:
    print(conversation.id, conversation.name)

files = run.output_files()
print([(output_file.path, output_file.size) for output_file in files])
image = run.download_file(files[0], "downloads/duck.png")
archive = run.download_all("downloads/all-results.tar")
```

The durable run ID is available immediately after admission:

```python
print(run.id)
```

Another process can attach to it later:

```python
client = ThinkClient()
run = client.resume("run-or-thread-id")
outcome = run.wait()
```

`run.messages` provides the hydrated transcript and `run.history` provides
bounded SDK progress history. Socket reconnects automatically replay the
current overlay state. When a replayed `needs_input` overlay arrives before its
transcript, `wait()` holds the outcome until transcript hydration restores the
clarification or plan-review prompt.

## Follow-up turns and errors

After a successful result, start another turn in the same conversation:

```python
run.follow_up("Now stratify the result by ancestry")
next_result = run.wait()
```

Transport, HTTP, billing, admission, timeout, and protocol failures have typed
exceptions under `bystro.think`. A `ThinkClient` owns one foreground run at a
time; use separate clients for concurrently controlled conversations. Closing
a client disconnects local transports but does not cancel durable server work.
