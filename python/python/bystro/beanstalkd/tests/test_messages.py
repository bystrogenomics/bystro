import unittest
from enum import Enum

from msgspec import Struct, json
from msgspec.inspect import type_info, Field, IntType, NODEFAULT

from bystro.beanstalkd.messages import (
    SubmissionID,
    BeanstalkJobID,
    Event,
    BaseMessage,
    SubmittedJobMessage,
    CompletedJobMessage,
    FailedJobMessage,
    InvalidJobMessage,
    ProgressData,
    ProgressMessage,
)


class T(Struct):
    a: int
    b: str


class ImmutableT(Struct, frozen=True):
    a: int
    b: str


class DefaultT(Struct, frozen=True):
    a: int
    b: str = "test"


class InheritedT(T):
    c: str


class ModuleTests(unittest.TestCase):
    def test_struct(self):
        t = T(a=1, b="test")
        self.assertEqual(t.a, 1)
        self.assertEqual(t.b, "test")

        t_immutable = ImmutableT(a=1, b="test")
        with self.assertRaisesRegex(AttributeError, "immutable type: 'ImmutableT'"):
            t_immutable.a = 2  # type: ignore

        t_default = DefaultT(a=1)
        self.assertEqual(t_default.b, "test")

        with self.assertRaisesRegex(TypeError, "Missing required argument 'c'"):
            InheritedT(a=5, b="test")  # type: ignore

    def test_event_enum(self):
        self.assertTrue(issubclass(Event, Enum))
        self.assertEqual(Event.PROGRESS, "progress")
        self.assertEqual(Event.FAILED, "failed")
        self.assertEqual(Event.STARTED, "started")
        self.assertEqual(Event.COMPLETED, "completed")

    def test_base_message(self):
        self.assertTrue(issubclass(BaseMessage, Struct))

        t = BaseMessage(submission_id="test")
        types = BaseMessage.keys_with_types()

        self.assertEqual(t.submission_id, "test")
        self.assertEqual(set(types.keys()), set({"submission_id"}))
        self.assertEqual(types["submission_id"], SubmissionID)

        with self.assertRaisesRegex(AttributeError, "immutable type: 'BaseMessage'"):
            t.submission_id = "test2"  # type: ignore

    def test_submitted_job_message(self):
        self.assertTrue(issubclass(SubmittedJobMessage, BaseMessage))

        t = SubmittedJobMessage(submission_id="test")
        types = SubmittedJobMessage.keys_with_types()

        self.assertEqual(t.event, Event.STARTED)
        self.assertEqual(set(types.keys()), set({"submission_id", "event"}))
        self.assertEqual(types["submission_id"], SubmissionID)
        self.assertEqual(types["event"], Event)

        with self.assertRaisesRegex(AttributeError, "immutable type: 'SubmittedJobMessage'"):
            t.submission_id = "test2"  # type: ignore

        with self.assertRaisesRegex(AttributeError, "immutable type: 'SubmittedJobMessage'"):
            t.event = Event.PROGRESS  # type: ignore

    def test_completed_job_message(self):
        self.assertTrue(issubclass(CompletedJobMessage, BaseMessage))

        t = CompletedJobMessage(submission_id="test")
        types = CompletedJobMessage.keys_with_types()

        self.assertEqual(t.event, Event.COMPLETED)
        self.assertEqual(set(types.keys()), set({"submission_id", "event"}))
        self.assertEqual(types["submission_id"], SubmissionID)
        self.assertEqual(types["event"], Event)

        with self.assertRaisesRegex(AttributeError, "immutable type: 'CompletedJobMessage'"):
            t.submission_id = "test2"  # type: ignore

        with self.assertRaisesRegex(AttributeError, "immutable type: 'CompletedJobMessage'"):
            t.event = Event.PROGRESS  # type: ignore

    def test_failed_job_message(self):
        self.assertTrue(issubclass(FailedJobMessage, BaseMessage))

        with self.assertRaisesRegex(TypeError, "Missing required argument 'reason'"):
            t = FailedJobMessage(submission_id="test")  # type: ignore

        t = FailedJobMessage(submission_id="test", reason="foo")
        types = FailedJobMessage.keys_with_types()

        self.assertEqual(t.event, Event.FAILED)
        self.assertEqual(set(types.keys()), set({"submission_id", "event", "reason"}))
        self.assertEqual(types["submission_id"], SubmissionID)
        self.assertEqual(types["event"], Event)
        self.assertEqual(types["reason"], str)

        with self.assertRaisesRegex(AttributeError, "immutable type: 'FailedJobMessage'"):
            t.event = "test2"  # type: ignore

    def test_invalid_job_message(self):
        self.assertTrue(issubclass(InvalidJobMessage, Struct))
        self.assertTrue(not issubclass(InvalidJobMessage, BaseMessage))

        with self.assertRaisesRegex(TypeError, "Missing required argument 'reason'"):
            t = InvalidJobMessage(queue_id="test")  # type: ignore

        t = InvalidJobMessage(queue_id=1, reason="foo")
        types = InvalidJobMessage.keys_with_types()

        self.assertEqual(t.event, Event.FAILED)
        self.assertEqual(set(types.keys()), set({"queue_id", "event", "reason"}))
        self.assertEqual(types["queue_id"], BeanstalkJobID)
        self.assertEqual(types["event"], Event)
        self.assertEqual(types["reason"], str)

        with self.assertRaisesRegex(AttributeError, "immutable type: 'InvalidJobMessage'"):
            t.event = "test2"  # type: ignore

    def test_progress_data(self):
        self.assertTrue(issubclass(ProgressData, Struct))

        types = list(type_info(ProgressData).fields)  # type: ignore

        expected_types = [
            Field(
                name="progress",
                encode_name="progress",
                type=IntType(gt=None, ge=None, lt=None, le=None, multiple_of=None),
                required=False,
                default=0,
                default_factory=NODEFAULT,
            ),
            Field(
                name="skipped",
                encode_name="skipped",
                type=IntType(gt=None, ge=None, lt=None, le=None, multiple_of=None),
                required=False,
                default=0,
                default_factory=NODEFAULT,
            ),
        ]
        self.assertListEqual(types, expected_types)

        t = ProgressData()
        t.progress = 1
        t.skipped = 2

        self.assertEqual(t.progress, 1)
        self.assertEqual(t.skipped, 2)

    def test_progress_message(self):
        self.assertTrue(issubclass(ProgressMessage, BaseMessage))

        with self.assertRaisesRegex(TypeError, "Missing required argument 'submission_id'"):
            t = ProgressMessage()  # type: ignore

        t = ProgressMessage(submission_id="test")
        types = ProgressMessage.keys_with_types()

        self.assertEqual(t.event, Event.PROGRESS)
        self.assertEqual(set(types.keys()), set({"submission_id", "event", "data"}))
        self.assertEqual(types["submission_id"], SubmissionID)
        self.assertEqual(types["event"], Event)
        self.assertEqual(types["data"], ProgressData)

        with self.assertRaisesRegex(AttributeError, "immutable type: 'ProgressMessage'"):
            t.submission_id = "test2"  # type: ignore

        t.data.progress = 1
        t.data.skipped = 2

        self.assertEqual(t.data.progress, 1)
        self.assertEqual(t.data.skipped, 2)

    def test_job_started_message_camel_decamel(self):
        msg = SubmittedJobMessage(submission_id="my_submission_id2")

        serialized_values = json.encode(msg)
        expected_value = {"submissionId": "my_submission_id2", "event": "started"}
        serialized_expected_value = json.encode(expected_value)

        self.assertEqual(serialized_values, serialized_expected_value)

        deserialized_values = json.decode(serialized_expected_value, type=SubmittedJobMessage)

        self.assertEqual(deserialized_values, msg)

    def test_job_completed_message_camel_decamel(self):
        msg = CompletedJobMessage(submission_id="my_submission_id2")

        serialized_values = json.encode(msg)
        expected_value = {"submissionId": "my_submission_id2", "event": "completed"}
        serialized_expected_value = json.encode(expected_value)

        self.assertEqual(serialized_values, serialized_expected_value)

        deserialized_values = json.decode(serialized_expected_value, type=CompletedJobMessage)

        self.assertEqual(deserialized_values, msg)

    def test_job_failed_message_camel_decamel(self):
        msg = FailedJobMessage(submission_id="my_submission_id2", reason="foo")

        serialized_values = json.encode(msg)
        expected_value = {"submissionId": "my_submission_id2", "reason": "foo", "event": "failed"}
        serialized_expected_value = json.encode(expected_value)

        self.assertEqual(serialized_values, serialized_expected_value)

        deserialized_values = json.decode(serialized_expected_value, type=FailedJobMessage)

        self.assertEqual(deserialized_values, msg)

    def test_invalid_job_message_camel_decamel(self):
        msg = InvalidJobMessage(queue_id=1, reason="foo")

        serialized_values = json.encode(msg)
        expected_value = {"queueId": 1, "reason": "foo", "event": "failed"}
        serialized_expected_value = json.encode(expected_value)

        self.assertEqual(serialized_values, serialized_expected_value)

        deserialized_values = json.decode(serialized_expected_value, type=InvalidJobMessage)

        self.assertEqual(deserialized_values, msg)

    def test_progress_message_camel_decamel(self):
        msg = ProgressMessage(
            submission_id="my_submission_id2", data=ProgressData(progress=1, skipped=2)
        )

        serialized_values = json.encode(msg)
        expected_value = {
            "submissionId": "my_submission_id2",
            "event": "progress",
            "data": {"progress": 1, "skipped": 2},
        }
        serialized_expected_value = json.encode(expected_value)

        self.assertEqual(serialized_values, serialized_expected_value)

        deserialized_values = json.decode(serialized_expected_value, type=ProgressMessage)

        self.assertEqual(deserialized_values, msg)

    def test_base_message_camel_decamel(self):
        msg = BaseMessage(submission_id="my_submission_id2")

        serialized_values = json.encode(msg)
        expected_value = {"submissionId": "my_submission_id2"}
        serialized_expected_value = json.encode(expected_value)

        self.assertEqual(serialized_values, serialized_expected_value)

        deserialized_values = json.decode(serialized_expected_value, type=BaseMessage)

        self.assertEqual(deserialized_values, msg)


if __name__ == "__main__":
    unittest.main()
