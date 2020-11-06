"""
FDET-Averaged
Tools for FDET with MD averaged rhoB densities.

"""
import os
import re
import sys
import sysconfig
import platform
import subprocess

from distutils.version import LooseVersion
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from setuptools import find_packages
import versioneer


short_description = __doc__.split("\n")
# from https://github.com/pytest-dev/pytest-runner#conditional-requirement
needs_pytest = {'pytest', 'test', 'ptr'}.intersection(sys.argv)
pytest_runner = ['pytest-runner'] if needs_pytest else []
os.environ['FDETAPATH'] = '$PWD'
os.environ['FDETADATA'] = '$FDETAPATH/fdeta/data'

try:
    with open("README.md", "r") as handle:
        long_description = handle.read()
except:
    long_description = "\n".join(short_description[2:]),


# As of Python 3.6, CCompiler has a `has_flag` method.
# cf http://bugs.python.org/issue26689
def has_flag(compiler, flagname):
    """Return a boolean indicating whether a flag name is supported on
    the specified compiler.
    """
    import tempfile
    with tempfile.NamedTemporaryFile('w', suffix='.cpp') as f:
        f.write('int main (int argc, char **argv) { return 0; }')
        try:
            compiler.compile([f.name], extra_postargs=[flagname])
        except setuptools.distutils.errors.CompileError:
            return False
    return True


# For testing
from setuptools.command.test import test as TestCommand

class CatchTestCommand(TestCommand):
    """
    A custom test runner to execute both Python unittest tests and C++ Catch-
    lib tests.
    """
    def distutils_dir_name(self, dname):
        """Returns the name of a distutils build directory"""
        dir_name = "{dirname}.{platform}-{version[0]}.{version[1]}"
        return dir_name.format(dirname=dname,
                               platform=sysconfig.get_platform(),
                               version=sys.version_info)

    def run(self):
        # Run Python tests
        super(CatchTestCommand, self).run()

setup(
    name='fdeta',
    author='Cristina E. González-Espinoza',
    author_email='crisbeth46@gmail.com',
    description=short_description[0],
    long_description=long_description,
    long_description_content_type="text/markdown",
    version=versioneer.get_version(),
    license='LGPLv3',
    packages=find_packages(),
    include_package_data=True,
    # Allows `setup.py test` to work correctly with pytest
#   install_requires=['pyclustering>=0.10.0', 
#                     'qcelemental>=0.4.0',
#                     'chemcoord>=2.0.4'],
    setup_requires=pytest_runner,
    # Additional entries
    python_requires=">=3.6",
    classifiers=[
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'Operating System :: Linux',
        'Programming Language :: Python :: 3',
        'Topic :: Science :: Quantum Chemistry :: Multidimensional Methods'
    ],
)
