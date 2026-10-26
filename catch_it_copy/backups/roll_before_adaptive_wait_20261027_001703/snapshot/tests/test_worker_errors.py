import unittest
import io
from contextlib import redirect_stderr
from gym_dcmm.utils.worker_errors import report_worker_error


class WorkerErrors(unittest.TestCase):
    def test_original_exception_printed_and_reraised(self):
        @report_worker_error
        def step(self):
            raise ValueError('original simulator failure')
        stream = io.StringIO()
        with redirect_stderr(stream), self.assertRaisesRegex(ValueError, 'original simulator failure'):
            step(self)
        self.assertIn('[environment-error]', stream.getvalue())
        self.assertIn('Traceback', stream.getvalue())

    def test_result_preserved(self):
        @report_worker_error
        def step(self):
            return (1, 2, False, False, {})
        self.assertEqual(step(self), (1, 2, False, False, {}))
