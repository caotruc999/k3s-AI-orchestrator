import os
import sys

MODULES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "modules")
if MODULES_DIR not in sys.path:
    sys.path.insert(0, MODULES_DIR)
