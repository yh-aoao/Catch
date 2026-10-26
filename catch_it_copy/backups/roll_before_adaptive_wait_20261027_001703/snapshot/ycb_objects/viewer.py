"""Keep the passive viewer's GUI thread alive through orderly shutdown."""
from contextlib import contextmanager
import threading


@contextmanager
def passive_viewer(model, data):
    from mujoco import viewer
    before = set(threading.enumerate())
    window = viewer.launch_passive(model, data)
    threads = [thread for thread in threading.enumerate()
               if thread not in before and '_launch_internal' in thread.name]
    try:
        yield window
    finally:
        window.close()
        # Handle.close() only requests exit. Wait before Python/GL teardown.
        for thread in threads:
            thread.join(timeout=5.)
            if thread.is_alive():
                raise RuntimeError('MuJoCo viewer did not finish shutting down')
