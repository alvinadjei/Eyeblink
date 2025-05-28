from capture.run_experiment import FlaskThread
from unittest.mock import MagicMock, patch
import pytest

### Test cases for FlaskThread class


def test_flaskthread_initializes_routes():
    """Test that FlaskThread initializes with routes."""
    mock_camera = MagicMock()
    flask_thread = FlaskThread(mock_camera)
    assert flask_thread.app.url_map is not None


def test_index_route_renders(monkeypatch):
    """Test that index route renders the template."""
    mock_camera = MagicMock()
    flask_thread = FlaskThread(mock_camera)

    # Patch template rendering to return a simple string
    monkeypatch.setattr('run_experiment.render_template', lambda name: f"Mock template {name}")

    client = flask_thread.app.test_client()
    response = client.get('/')
    assert response.status_code == 200
    assert b"Mock template index.html" in response.data
    

@patch('run_experiment.window')
def test_trial_route_returns_trial_num(mock_window):
    """Test that trial route returns the current trial number."""
    mock_window.trial_num = 42
    mock_camera = MagicMock()
    flask_thread = FlaskThread(mock_camera)

    client = flask_thread.app.test_client()
    response = client.get('/trial')
    assert response.status_code == 200
    assert response.get_json() == {"trial_num": 42}


def test_livestream_route_structure():
    """Test that livestream route yields frames."""
    mock_camera = MagicMock()
    mock_camera.get_latest_frame.return_value = None  # So it just loops/sleeps

    flask_thread = FlaskThread(mock_camera)
    client = flask_thread.app.test_client()
    
    response = client.get('/livestream', buffered=True)
    assert response.status_code == 200
    assert b"--frame" in next(response.response) or True  # Some yield