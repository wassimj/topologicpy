"""Run backend-independent API checks in isolation from the real kernel tests."""
from pathlib import Path
import subprocess
import sys


def test_public_modifier_contract_in_isolated_process():
    script = Path(__file__).with_name("modifier_provenance_contract_runner.py")
    completed = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stdout + completed.stderr
