from capture.run_experiment import MainWindow
import pytest
from unittest.mock import MagicMock
from PyQt5.QtWidgets import QApplication

### Test cases for MainWindow class


def test_mainwindow_initializes(qtbot):
    """Test that the MainWindow initializes with the correct title and components."""
    window = MainWindow()
    qtbot.addWidget(window)

    assert window.windowTitle() == "FEC Measurement"
    assert window.trial_num == 1
    assert window.start_button.text() == "Start Experiment"
    assert window.video_label is not None
    assert window.explanation_label is not None
    

def test_start_button_triggers_start_experiment(qtbot):
    """Test that clicking the start button calls the start_experiment method."""
    window = MainWindow()
    qtbot.addWidget(window)

    window.start_experiment = MagicMock()
    window.start_button.clicked.emit()  # Simulate button click

    window.start_experiment.assert_called_once()


def test_threads_are_initialized(qtbot):
    """Test that the camera and Flask threads are initialized but not running."""
    window = MainWindow()
    qtbot.addWidget(window)

    assert window.camera_thread is not None
    assert window.flask_thread is not None
    assert window.camera_thread.isRunning() is False  # Not started yet
    assert window.flask_thread.isRunning() is False


def test_explanation_label_text(qtbot):
    """Test that the explanation label contains the expected text."""
    window = MainWindow()
    qtbot.addWidget(window)

    assert "This experiment involves combining a conditioned stimulus" in window.explanation_label.text()
    assert "Mouse controls:" in window.explanation_label.text()
    assert "Key controls" in window.explanation_label.text()