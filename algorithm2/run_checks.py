"""Run the offline test suite from the algorithm-2 directory."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    return subprocess.call([sys.executable, "-m", "pytest", "-q", "tests"], cwd=ROOT)


if __name__ == "__main__":
    sys.exit(main())
