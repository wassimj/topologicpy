try:
    import openstudio
    import honeybee_energy
except ImportError as e:
    print("honeybee-energy is not installed. Please install it to run this test.")
    raise e
    exit(1)

def test_set_energy_model_log_level():
    # Not implemented yet
    import openstudio
    try:
        openstudio.Logger.instance().standardOutLogger().setLogLevel(openstudio.Fatal)
        assert True
    except Exception as e:
        print("Failed to set OpenStudio log level.")
        assert False
        
