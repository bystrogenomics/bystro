# Batch Submission API

Bystro uses [Beanstalkd](https://beanstalkd.github.io) for batch submissions, for both Perl and Python APIs. In this document we describe how this works, in the context of the Python API.

Briefly, Beanstalkd is a simple pub-sub protocol that features persistence and which has the performance of Redis. However, unlike Redis, it has useful features that make it approprirate for a job queue:

- Exactly once delivery semantics: A message can be reserved by only a single beanstalkd client at a time.
- TTR: If a message is reserved for too long, it will be released back for further reservation, to guard against workers that lose connectivity

Like Redis, it features multiple tubes/channels, which we can use to differentiate between job categories/types.

## Submitting a job

Every Bystro job must be defined in the beanstalkd config, [config/beanstalk.clean.yml](https://github.com/bystrogenomics/bystro/blob/master/config/beanstalk.clean.yml). When Bystro is deployed on a server, this file should be renamed to `beanstalk.yml``, and the address of the beanstalk must be filled under the addresses field (more than 1 address is allowed):

Here is the structure of the beanstalkd.clean.yml:

```yaml
beanstalkd:
  addresses:
    - <host1>:<port1>
```

- `addresses` is a list of host:port values (e.g bystro-dev.emory.edu:11300)

```yaml
tubes:
  annotation:
    submission: annotation
    events: annotation_events
```

- `tubes` defines a dictionary of objects, each of which defines 2 tubes for a single job category.type:
  - The object is composed of a `submission` tube and an `events` tube.
    - submission: <submission tube name>
      - This is the tube that the producer uses to push a new job, that a worker will listen on. Upon picking up a new job message in this tube, the worker will do some processing on the job.
    - events: <events tube name>
      - This is the tube that workers/consumers will push messages about the state of the job, and the final output of the job to. The producer will listen on this tube for updates about the job they earlier submitted on the `submission` tube.

Let's take an example from the <b>bystro.search.index</b> python library:

## Indexing Python Library Example

We will focus on `bystro.search.index.listener`, which defines how job submissions are processed, and how we notify the web backend that a job has been received for processing, has successfully completed, or has failed.

### handler_fn

The handler_fn, or handler function, defines how the job is handled.

```python
def handler_fn(publisher: ProgressPublisher, beanstalkd_job_data: IndexJobData):
    m_path = get_config_file_path(conf_dir, beanstalkd_job_data.assembly, ".mapping.y*ml")

    with open(m_path, "r", encoding="utf-8") as f:
        mapping_conf = YAML(typ="safe").load(f)

    inputs = beanstalkd_job_data.inputFileNames

    if not inputs.archived:
        raise ValueError("Indexing currently only works for indexing archived (tarballed) results")

    tar_path = os.path.join(beanstalkd_job_data.inputDir, inputs.archived)

    return asyncio.get_event_loop().run_until_complete(go(
        index_name=beanstalkd_job_data.indexName,
        tar_path=tar_path,
        mapping_conf=mapping_conf,
        search_conf=search_conf,
        publisher=publisher,
    ))
```

The handler_fn takes a ` ProgressPublisher``, which is configured to publish event messages to the this job's  `event``tube, and`IndexJobData`, which is the `msgspec.Struct` that defines the JSON message the worker expects.

Let's take a look at [IndexJobData](https://github.com/bystrogenomics/bystro/blob/91934b83002694f46e34b0317fa398441e4293ed/python/python/bystro/search/utils/messages.py#L5):

```python
class IndexJobData(BaseMessage, frozen=True):
    """Data for SaveFromQuery jobs received from beanstalkd"""

    inputDir: str
    inputFileNames: AnnotationOutputs
    indexName: str
    assembly: str
    fieldNames: list[str] | None = None
    indexConfig: dict | None = None
```

This inherits the properties from `BaseMessage`, and a number of properties specific to search indexing, like `inputData`. [BaseMessage](https://github.com/bystrogenomics/bystro/blob/91934b83002694f46e34b0317fa398441e4293ed/python/python/bystro/beanstalkd/messages.py#L17) looks like this:

```python
class BaseMessage(Struct, frozen=True):
    submissionID: SubmissionID

    @classmethod
    def keys_with_types(cls) -> dict:
        return get_type_hints(cls)
```

It defines a single property, submissionID. The submissionID is the primary key / id of the job submission. In the Bystro webapp backend, we store 1 ID per submission, which allows us to re-try a given job multiple times, while still recording the unique events associated with each submission independently.

### submit_msg_fn

https://github.com/bystrogenomics/bystro/blob/master/python/python/bystro/search/index/listener.py#L71

The [submit_msg_fn](https://github.com/bystrogenomics/bystro/blob/master/python/python/bystro/search/index/listener.py#L71), or submit message function, defines what message is sent on this job types `events` tube upon successfully receiving a job message, parsing it's json, and validating that the JSON is of the expected shape/type (here `IndexJobData` as mentioned above)

It is quite simple:

```yaml
def submit_msg_fn(job_data: IndexJobData):
    return SubmittedJobMessage(job_data.submissionID)
```

Where [SubmittedJobMessage](https://github.com/bystrogenomics/bystro/blob/91934b83002694f46e34b0317fa398441e4293ed/python/python/bystro/beanstalkd/messages.py#L24) is:

```python
class SubmittedJobMessage(BaseMessage, frozen=True):
    event: Event = Event.STARTED
```

Like all other messages it extends from BaseMessage, and so has the `submissionID` property as well, which is used to indicate to the submitter (the web server) which job this message pertains to. You'll also notice that it has an `event` property. Every `events` tube message should have an `event` property, which must have value `progress`, `failed`, `started`, or `completed`, as defined in [Event](https://github.com/bystrogenomics/bystro/blob/91934b83002694f46e34b0317fa398441e4293ed/python/python/bystro/beanstalkd/messages.py#L9C1-L15C28):

```python
class Event(str, Enum):
    """Beanstalkd Event"""

    PROGRESS = "progress"
    FAILED = "failed"
    STARTED = "started"
    COMPLETED = "completed"
```

The failed_msg_fn, and completed_msg_fn are similarly structured.
