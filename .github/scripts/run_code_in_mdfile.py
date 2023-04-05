#!/usr/bin/env python

import argparse
import subprocess
import tempfile

parser = argparse.ArgumentParser(description='Run all the code for a given language in an MD file')
parser.add_argument('language')
parser.add_argument('file')
args = parser.parse_args()

def enumerate_string_lines(code):
    i=1
    print("```")
    for line in code.split("\n"):
        print(f"{i:>3} {line}")
        i+=1
    print("```")

def process_file(file, language):
    with open(file) as fh:
        result=subprocess.run(['codedown', language + '*'], stdin=fh, capture_output=True, text=True)

        code = result.stdout.strip()

        if len(code) == 0:
            return 0

        if args.language == 'python':
            enumerate_string_lines(code)
            code_result=subprocess.run('python3', input=code, text=True)
            exit(code_result.returncode)
        elif args.language == 'rust':
            rust_script_contents=f"""//! ```cargo
        //! [dependencies]
        //! lace = {{ path = ".", version="0.1.0" }}
        //! ```
        fn main() {{
        {code}
        }}
        """

            enumerate_string_lines(rust_script_contents)

            rust_script_file=tempfile.NamedTemporaryFile(mode='w', dir='./lace')
            rust_script_file.write(rust_script_contents)
            rust_script_file.flush()

            code_result=subprocess.run(['rust-script', rust_script_file.name])
            return code_result.returncode
        else:
            return 1

retval=process_file(args.file, args.language)
exit(retval)
