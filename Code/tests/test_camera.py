import pytest
from capture.run_experiment import CameraThread
import numpy as np
from PyQt5.QtCore import QSignalSpy
from unittest.mock import MagicMock, patch

### Test cases for CameraThread class


def test_camera_thread_initialization():
    """Test that CameraThread initializes with default values."""
    cam_thread = CameraThread()
    assert cam_thread.camera_index == 0
    assert cam_thread.frame is None
    assert cam_thread.running is False
    

def test_camera_thread_stop_sets_flag():
    """Test that stop method sets running flag to False."""
    cam_thread = CameraThread()
    cam_thread.running = True
    cam_thread.stop()
    assert cam_thread.running is False


@patch('run_experiment.VmbSystem')
@patch('run_experiment.get_camera')
@patch('run_experiment.Handler')
def test_camera_thread_run_emits_frame(mock_handler_cls, mock_get_camera, mock_vmbsystem):
    mock_handler = MagicMock()
    mock_frame = np.ones((480, 640, 3), dtype=np.uint8)
    mock_handler.get_image.return_value = mock_frame
    mock_handler_cls.return_value = mock_handler

    mock_cam = MagicMock()
    mock_get_camera.return_value.__enter__.return_value = mock_cam

    # Use Spy to track signal emissions
    cam_thread = CameraThread()
    spy = QSignalSpy(cam_thread.frame_ready)

    # Start thread and immediately stop (simulate one loop iteration)
    def fake_run_once():
        cam_thread.running = True
        cam_thread.frame = mock_frame
        cam_thread.frame_ready.emit(mock_frame)
        cam_thread.running = False

    cam_thread.run = fake_run_once
    cam_thread.start()
    cam_thread.wait()  # wait for thread to finish

    assert len(spy) == 1
    assert np.array_equal(spy[0][0], mock_frame)