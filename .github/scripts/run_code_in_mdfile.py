#!/usr/bin/env python

import argparse
import subprocess

parser = argparse.ArgumentParser(description='Run all the code for a given language in an MD file')
parser.add_argument('language')
parser.add_argument('file')
args = parser.parse_args()

fh = open(args.file)

result=subprocess.run(['codedown', args.language], stdin=fh, capture_output=True, text=True)

code = result.stdout.strip()

if len(code) == 0:
    exit(0)

exit(1)
