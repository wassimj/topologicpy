"""Execute the tutorial suite in isolated, fresh Jupyter kernels."""
import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import tempfile
import time

import nbformat
from nbclient import NotebookClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--only', nargs='+', help='Two-digit tutorial numbers, e.g. 09 11')
    parser.add_argument('--output', type=Path, help='JSON report path')
    options = parser.parse_args()
    root = Path(__file__).resolve().parent
    paths = sorted(root.glob('[0-9][0-9]_*.ipynb'))
    if options.only:
        paths = [p for p in paths if p.name[:2] in options.only]
    if not paths:
        parser.error('No matching tutorial notebooks')
    report = {'python': platform.python_version(), 'platform': platform.platform(),
              'interpreter': sys.executable, 'notebooks': []}
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    with tempfile.TemporaryDirectory(prefix='topologicpy_verify_') as temporary:
        temporary = Path(temporary)
        kernel = temporary / 'jupyter' / 'kernels' / 'tutorial-validation'
        kernel.mkdir(parents=True)
        (kernel / 'kernel.json').write_text(json.dumps({
            'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
            'display_name': 'Tutorial validation', 'language': 'python'}), encoding='utf-8')
        os.environ['JUPYTER_PATH'] = str(temporary / 'jupyter')
        os.environ['IPYTHONDIR'] = str(temporary / 'ipython')
        os.environ['JUPYTER_RUNTIME_DIR'] = str(temporary / 'runtime')
        Path(os.environ['IPYTHONDIR']).mkdir()
        Path(os.environ['JUPYTER_RUNTIME_DIR']).mkdir()
        os.environ['TOPOLOGICPY_CORE_BACKEND'] = 'pythonocc'
        for path in paths:
            folder = temporary / path.stem
            folder.mkdir()
            shutil.copytree(root / 'data', folder / 'data')
            notebook = nbformat.read(path, as_version=4)
            nbformat.validate(notebook)
            result = {'notebook': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                      'status': 'passed'}
            print('Running', path.name, flush=True)
            start = time.perf_counter()
            try:
                NotebookClient(notebook, kernel_name='tutorial-validation', timeout=120,
                               startup_timeout=120, record_timing=True,
                               resources={'metadata': {'path': str(folder)}}).execute()
            except Exception as error:
                result['status'] = 'failed'
                result['error'] = str(error)
            result['seconds'] = round(time.perf_counter() - start, 2)
            report['notebooks'].append(result)
            print(result['status'], result['seconds'], 'seconds', flush=True)
    output = options.output or root / 'verification-local.json'
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('Report:', output)
    return int(any(r['status'] != 'passed' for r in report['notebooks']))


if __name__ == '__main__':
    sys.exit(main())
