import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app import JobManager
import queue

def test_job_manager_initial_state():
    jm = JobManager()
    assert jm.job_active is False
    assert jm.output_file == ""
    assert not jm.cancel_event.is_set()

def test_job_manager_reset():
    jm = JobManager()
    jm.progress_q.put({"test": "data"})
    jm.cancel_event.set()

    jm.reset_for_new_job("new_output.mp4")

    assert jm.progress_q.empty()
    assert not jm.cancel_event.is_set()
    assert jm.output_file == "new_output.mp4"
