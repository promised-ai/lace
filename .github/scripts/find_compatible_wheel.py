#!/usr/bin/env python

import argparse
from packaging.tags import sys_tags
import os.path
import glob
import sys

parser = argparse.ArgumentParser(
    description="Program to find wheels in a directory compatible with the current version of Python"
)

parser.add_argument("package", help="The name of the package that you are searching for a wheel for")
parser.add_argument("dir", help="the directory under which to search for the wheels")

args=parser.parse_args()

wheel=None

for tag in sys_tags():
    matches=glob.glob(args.package + "*" + str(tag) + ".whl", root_dir=args.dir)
    if len(matches) == 1:
        wheel=matches[0]
        break
    elif len(matches) > 1:
        print("Found multiple matches for the same tag " + str(tag), matches, file=sys.stderr)

if wheel: 
    print(os.path.join(args.dir, wheel))
else:
    sys.exit("Did not find compatible wheel")
