from importlib.resources import files
from pathlib import Path

PACKAGE_NAME = "caafe"
PACKAGE_PATH = str(Path(files(PACKAGE_NAME)))
