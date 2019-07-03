"""
Unit and regression test for the fdeta package.
"""

# Import package, test suite, and other packages as needed
import fdeta
import pytest
import sys

def test_fdeta_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "fdeta" in sys.modules
