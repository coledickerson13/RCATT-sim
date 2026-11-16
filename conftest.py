"""pytest conftest — adds the rcatt_sim root to sys.path."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
