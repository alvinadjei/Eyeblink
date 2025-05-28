from capture.run_experiment import MainWindow

def test_initial_trial_number(qtbot):
    window = MainWindow()
    assert window.trial_num == 1
