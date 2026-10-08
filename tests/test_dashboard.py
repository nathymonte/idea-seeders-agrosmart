import io
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from scripts.demo import build_demo
from services.paths import PROJECT_ROOT


class DashboardTest(unittest.TestCase):
    def run_app(self, lake: Path, images: Path, assertions: str):
        # Separate processes ensure environment-configured paths and Streamlit caches are isolated.
        code = "from streamlit.testing.v1 import AppTest; "
        code += f"app=AppTest.from_file({str(PROJECT_ROOT / 'dashboard/app.py')!r},default_timeout=30).run(); "
        code += "assert not app.exception, str(app.exception); assert not app.error, str(app.error); " + assertions
        result = subprocess.run([sys.executable, '-B', '-c', code], cwd=PROJECT_ROOT,
                                env={**os.environ, 'DATA_LAKE_PATH': str(lake), 'AGROSMART_IMAGE_CSV': str(images)},
                                capture_output=True, text=True, timeout=90)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_empty_dashboard(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.run_app(Path(tmp) / 'lake', Path(tmp) / 'absent.csv', 'assert len(app.info) >= 2')

    def test_demo_filters_and_tabs(self):
        with tempfile.TemporaryDirectory() as tmp:
            with redirect_stdout(io.StringIO()):
                build_demo(Path(tmp))
            self.run_app(Path(tmp) / 'data_lake', Path(tmp) / 'output/classificacoes.csv',
                         "assert len(app.tabs)==3; app.sidebar.selectbox[0].select('TALHAO-02').run(); "
                         "assert not app.exception; assert not app.error; "
                         "assert any(m.label=='Umidade do solo' for m in app.metric); "
                         "assert any('Sem analises' in i.value for i in app.info)")

    def test_sensor_only_dashboard(self):
        with tempfile.TemporaryDirectory() as tmp:
            with redirect_stdout(io.StringIO()):
                build_demo(Path(tmp))
            self.run_app(Path(tmp) / 'data_lake', Path(tmp) / 'missing.csv', 'assert len(app.tabs)==3')
