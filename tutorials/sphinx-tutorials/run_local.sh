#!/bin/bash

set -e
set -v

# Allows you to run all the tutorials without building the docset.

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# loop through all the .py files in the directory
for file in $(ls -r "$DIR"/*.py)
do
  # execute each Python script using the 'exec' function
  echo $file
  python -c """
with open('$file') as f:
    source = f.read()
code = compile(source, '$file', 'exec')

exec(code)
"""
done
