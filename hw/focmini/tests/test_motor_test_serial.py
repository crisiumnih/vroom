"""Host-only serial failure injection; never opens a real device."""
import contextlib
import importlib.util
import io
from pathlib import Path
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location(
    "motor_test_serial", Path(__file__).resolve().parents[1] / "motor_test_serial.py")
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


class FakePort:
    def __init__(self, failure=None, acknowledgment=True):
        self.failure = failure
        self.acknowledgment = acknowledgment
        self.writes = []

    def write(self, data):
        self.writes.append(data)
        if data == b"RUN10\n":
            if self.failure == "write":
                raise KeyboardInterrupt
            if self.failure == "short":
                return 3
            if self.failure == "serial":
                raise helper.serial.SerialException("disconnected")
        if data == b"\nSTOP\n" and self.failure == "stop":
            raise OSError("port lost")
        return len(data)

    def flush(self):
        if self.writes == [b"RUN10\n"] and self.failure == "flush":
            raise KeyboardInterrupt

    def readline(self):
        if self.writes[-1] == b"\nSTOP\n":
            return b"STOP received: EN low.\n" if self.acknowledgment else b""
        if self.failure in {"read", "stop"}:
            raise KeyboardInterrupt
        return b"RUN10 complete: 10 s elapsed; EN low.\n"


class SerialCleanupTests(unittest.TestCase):
    def run_fake(self, port):
        output = io.StringIO()
        # Deterministic deadlines avoid real sleeps and hardware dependencies.
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output), \
                patch.object(helper.time, "monotonic", side_effect=range(100)):
            result = helper.run_once(port)
        return result, output.getvalue()

    def test_interrupt_during_write_flush_and_read_stops(self):
        for stage in ("write", "flush", "read"):
            with self.subTest(stage=stage):
                port = FakePort(stage)
                result, output = self.run_fake(port)
                self.assertEqual(result, 130)
                self.assertEqual(port.writes, [b"RUN10\n", b"\nSTOP\n"])
                self.assertIn("STOP acknowledged", output)
                self.assertNotIn("Cancelled before", output)

    def test_short_write_and_serial_error_attempt_stop(self):
        for stage in ("short", "serial"):
            with self.subTest(stage=stage):
                port = FakePort(stage)
                result, _ = self.run_fake(port)
                self.assertEqual(result, 1)
                self.assertEqual(port.writes[-1], b"\nSTOP\n")

    def test_unacknowledged_stop_is_reported(self):
        result, output = self.run_fake(FakePort("read", acknowledgment=False))
        self.assertEqual(result, 130)
        self.assertIn("not acknowledged", output)

    def test_cleanup_error_does_not_hide_interrupt(self):
        result, output = self.run_fake(FakePort("stop"))
        self.assertEqual(result, 130)
        self.assertIn("STOP not confirmed", output)

    def test_completion_does_not_send_stop(self):
        port = FakePort()
        result, _ = self.run_fake(port)
        self.assertEqual(result, 0)
        self.assertEqual(port.writes, [b"RUN10\n"])

    def test_run_timeout_attempts_stop(self):
        port = FakePort()
        with patch.object(helper, "RUN_TIMEOUT_S", 0):
            result, output = self.run_fake(port)
        self.assertEqual(result, 1)
        self.assertEqual(port.writes[-1], b"\nSTOP\n")
        self.assertIn("STOP acknowledged", output)


if __name__ == "__main__":
    unittest.main()
