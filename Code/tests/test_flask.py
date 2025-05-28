from capture.run_experiment import FlaskThread
from unittest.mock import MagicMock

def test_flask_trial_route():
    fake_cam = MagicMock()
    flask_thread = FlaskThread(fake_cam)
    
    test_client = flask_thread.app.test_client()
    response = test_client.get('/trial')
    assert response.status_code == 200
    assert 'trial_num' in response.json
