import os, sys

# Ensure local src/ package path is available for all entrypoints
_here = os.path.dirname(__file__)
_src = os.path.abspath(os.path.join(_here, 'src'))
if _src not in sys.path:
    sys.path.insert(0, _src)


