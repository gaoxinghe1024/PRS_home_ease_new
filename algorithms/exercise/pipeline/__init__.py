"""Offline stages; importing the package also puts the serving package on sys.path."""
from . import common  # noqa: F401 - patches sys.path before any stage imports homeease
