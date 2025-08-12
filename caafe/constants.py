from importlib.resources import files
from pathlib import Path

PACKAGE_PATH = str(Path(files("caafe")))
