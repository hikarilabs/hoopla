"""
Add the root of the folder to enable module discovery at root level
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
