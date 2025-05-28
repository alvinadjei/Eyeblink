from capture.run_experiment import CameraThread
import numpy as np

def test_camera_thread_initialization():
    cam_thread = CameraThread()
    assert cam_thread.camera_index == 0
    assert cam_thread.frame is None
    assert cam_thread.running is False
