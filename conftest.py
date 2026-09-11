import os
import sys

# Ensure the project root is importable as a package root (models.*, data.*, utils.*)
# regardless of how/where pytest is invoked from.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
