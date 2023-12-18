import unittest
from enum import Enum

from msgspec import Struct
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
            t_immutable.a = 2 # type: ignore

        t_default = DefaultT(a=1)
        self.assertEqual(t_default.b, "test")

        with self.assertRaisesRegex(TypeError, "Missing required argument 'c'"):
            InheritedT(a=5, b="test") # type: ignore

    def test_event_enum(self):
        self.assertTrue(issubclass(Event, Enum))
        self.assertEqual(Event.PROGRESS, "progress")
        self.assertEqual(Event.FAILED, "failed")
        self.assertEqual(Event.STARTED, "started")
        self.assertEqual(Event.COMPLETED, "completed")

    def test_base_message(self):
        self.assertTrue(issubclass(BaseMessage, Struct))

        t = BaseMessage(submissionID="test")
        types = BaseMessage.keys_with_types()

        self.assertEqual(t.submissionID, "test")
        self.assertEqual(set(types.keys()), set({"submissionID"}))
        self.assertEqual(types["submissionID"], SubmissionID)

        with self.assertRaisesRegex(AttributeError, "immutable type: 'BaseMessage'"):
            t.submissionID = "test2" # type: ignore

    def test_submitted_job_message(self):
        self.assertTrue(issubclass(SubmittedJobMessage, BaseMessage))

        t = SubmittedJobMessage(submissionID="test")
        types = SubmittedJobMessage.keys_with_types()

        self.assertEqual(t.event, Event.STARTED)
        self.assertEqual(set(types.keys()), set({"submissionID", "event"}))
        self.assertEqual(types["submissionID"], SubmissionID)
        self.assertEqual(types["event"], Event)

        with self.assertRaisesRegex(AttributeError, "immutable type: 'SubmittedJobMessage'"):
            t.submissionID = "test2" # type: ignore

        with self.assertRaisesRegex(AttributeError, "immutable type: 'SubmittedJobMessage'"):
            t.event = Event.PROGRESS # type: ignore

    def test_completed_job_message(self):
        self.assertTrue(issubclass(CompletedJobMessage, BaseMessage))

        t = CompletedJobMessage(submissionID="test")
        types = CompletedJobMessage.keys_with_types()

        self.assertEqual(t.event, Event.COMPLETED)
        self.assertEqual(set(types.keys()), set({"submissionID", "event"}))
        self.assertEqual(types["submissionID"], SubmissionID)
        self.assertEqual(types["event"], Event)

        with self.assertRaisesRegex(AttributeError, "immutable type: 'CompletedJobMessage'"):
            t.submissionID = "test2" # type: ignore

        with self.assertRaisesRegex(AttributeError, "immutable type: 'CompletedJobMessage'"):
            t.event = Event.PROGRESS # type: ignore

    def test_failed_job_message(self):
        self.assertTrue(issubclass(FailedJobMessage, BaseMessage))

        with self.assertRaisesRegex(TypeError, "Missing required argument 'reason'"):
            t = FailedJobMessage(submissionID="test") # type: ignore

        t = FailedJobMessage(submissionID="test", reason="foo")
        types = FailedJobMessage.keys_with_types()

        self.assertEqual(t.event, Event.FAILED)
        self.assertEqual(set(types.keys()), set({"submissionID", "event", "reason"}))
        self.assertEqual(types["submissionID"], SubmissionID)
        self.assertEqual(types["event"], Event)
        self.assertEqual(types["reason"], str)

        with self.assertRaisesRegex(AttributeError, "immutable type: 'FailedJobMessage'"):
            t.event = "test2" # type: ignore

    def test_invalid_job_message(self):
        self.assertTrue(issubclass(InvalidJobMessage, Struct))
        self.assertTrue(not issubclass(InvalidJobMessage, BaseMessage))

        with self.assertRaisesRegex(TypeError, "Missing required argument 'reason'"):
            t = InvalidJobMessage(queueID="test") # type: ignore

        t = InvalidJobMessage(queueID=1, reason="foo")
        types = InvalidJobMessage.keys_with_types()

        self.assertEqual(t.event, Event.FAILED)
        self.assertEqual(set(types.keys()), set({"queueID", "event", "reason"}))
        self.assertEqual(types["queueID"], BeanstalkJobID)
        self.assertEqual(types["event"], Event)
        self.assertEqual(types["reason"], str)

        with self.assertRaisesRegex(AttributeError, "immutable type: 'InvalidJobMessage'"):
            t.event = "test2" # type: ignore

    def test_progress_data(self):
        self.assertTrue(issubclass(ProgressData, Struct))

        types = list(type_info(ProgressData).fields) # type: ignore

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

        with self.assertRaisesRegex(TypeError, "Missing required argument 'submissionID'"):
            t = ProgressMessage() # type: ignore

        t = ProgressMessage(submissionID="test")
        types = ProgressMessage.keys_with_types()

        self.assertEqual(t.event, Event.PROGRESS)
        self.assertEqual(set(types.keys()), set({"submissionID", "event", "data"}))
        self.assertEqual(types["submissionID"], SubmissionID)
        self.assertEqual(types["event"], Event)
        self.assertEqual(types["data"], ProgressData)

        with self.assertRaisesRegex(AttributeError, "immutable type: 'ProgressMessage'"):
            t.submissionID = "test2" # type: ignore

        t.data.progress = 1
        t.data.skipped = 2

        self.assertEqual(t.data.progress, 1)
        self.assertEqual(t.data.skipped, 2)


if __name__ == "__main__":
    unittest.main()
