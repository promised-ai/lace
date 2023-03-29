#!/usr/bin/env python

import argparse
import subprocess
import tempfile

parser = argparse.ArgumentParser(description='Run all the code for a given language in an MD file')
parser.add_argument('language')
parser.add_argument('file')
args = parser.parse_args()

fh = open(args.file)

result=subprocess.run(['codedown', args.language + '*'], stdin=fh, capture_output=True, text=True)

code = result.stdout.strip()

if len(code) == 0:
    exit(0)

if args.language == 'python':
    code_result=subprocess.run('python3', input=code, text=True)
    exit(code_result.returncode)
elif args.language == 'rust':
    rust_script_file=tempfile.NamedTemporaryFile(mode='w', dir='./lace')
    rust_script_file.write("""
//! ```cargo
//! [dependencies]
//! lace = { path = ".", version="0.1.0" }
//! ```

""")
    rust_script_file.write(code)
    rust_script_file.flush()

    code_result=subprocess.run(['rust-script', rust_script_file.name])
    exit(code_result.returncode)
else:
    exit(1)
