"""Asegura que la raíz del proyecto esté en sys.path para los imports `src.*`."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
