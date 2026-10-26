"""Print original simulator exceptions before old AsyncVectorEnv masks them."""
from functools import wraps
import os
import sys
import traceback


def report_worker_error(method):
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        except Exception:
            print('[environment-error] pid={} method={} motion={} step={}'.format(
                os.getpid(), method.__name__, getattr(self, 'object_motion', '?'),
                getattr(self, 'steps', '?')), file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)
            sys.stderr.flush()
            raise
    return wrapped
