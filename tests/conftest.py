"""
conftest.py — pytest configuration for rcatt_sim/tests/
Adds the rcatt_sim root to sys.path so `model.*` imports resolve.
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
