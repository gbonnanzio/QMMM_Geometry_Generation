#!/bin/bash

export BABEL_DATADIR="/mnt/c/Program Files (x86)/ADFRsuite-1.0/OpenBabel-2.4.1/data"
export PATH="/mnt/c/Program Files (x86)/ADFRsuite-1.0/bin:/mnt/c/Program Files (x86)/ADFRsuite-1.0/OpenBabel-2.4.1:$PATH"

"/mnt/c/Program Files (x86)/ADFRsuite-1.0/python.exe" "C:\Program Files (x86)\ADFRsuite-1.0\Lib\site-packages\AutoDockTools\Utilities24\prepare_receptor4.py" "$@"

