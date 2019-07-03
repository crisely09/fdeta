"""
FDET-Averaged
Tools for FDET with MD averaged rhoB densities.
"""

# Add imports here
from .fdeta import *

# Handle versioneer
from ._version import get_versions
versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions
