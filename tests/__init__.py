"""Deep Time regression tests."""
import sys
from pathlib import Path

# Many modules import the engine, and sixteen import sibling test modules, by bare name.
# That only worked for the latter once some other module had edited sys.path, so they
# could not be run on their own. This package is imported before any module in it.
_TESTS = Path(__file__).resolve().parent
for _path in (str(_TESTS.parent), str(_TESTS)):
    if _path not in sys.path:
        sys.path.insert(0, _path)
