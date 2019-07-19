#!/bin/bash
# Borrowed from HORTON 2 package
echo Cleaning python code in \'`pwd`\' and subdirectories
# split output of find at newlines.
IFS=$'\n'
# send all relevant files to the code cleaner
find fdeta tools *.py *.sh | egrep "(\.rst$)|(\.rst.template$)|(README)|(\.bib$)|(\.py$)|(\.c$)|(\.h$)|(\.nwchem)|(\.pyx$)|(\.pxd$)|(\.cpp)|(\.sh)|(\.cfg)|(\.gitignore)|(\.css)" | xargs ./tools/codecleaner.py
