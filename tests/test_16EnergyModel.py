import pytest


def test_set_energy_model_log_level():
    # Skip if openstudio is not installed
    openstudio = pytest.importorskip("openstudio", reason="openstudio is not installed")

    try:
        openstudio.Logger.instance().standardOutLogger().setLogLevel(openstudio.Fatal)
        assert True
    except Exception as e:
        print("Failed to set OpenStudio log level.")
        assert False
