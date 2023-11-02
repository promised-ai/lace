#!/usr/bin/env python

import argparse
import subprocess
import tempfile
import os


def print_filename(file_name):
    try:
        import termcolor

        display = termcolor.colored(file_name, "green")
        print(display)
    except ImportError:
        print(file_name)


def enumerate_string_lines(code):
    i = 1
    print("```")
    for line in code.split("\n"):
        print(f"{i:>3} {line}")
        i += 1
    print("```", flush=True)

def process_file(file, language, version):
    print_filename(file)
    with open(file) as fh:
        result = subprocess.run(
            ["codedown", language + "*"], stdin=fh, capture_output=True, text=True
        )

        code = result.stdout.strip()

        if len(code) == 0:
            return 0

        if args.language == "python":
            enumerate_string_lines(code)
            code_result = subprocess.run("python3", input=code, text=True)
            return code_result.returncode
        elif args.language == "rust":
            rust_script_contents = f"""//! ```cargo
        //! [dependencies]
        //! lace = {{ path = ".", version="{version}", features = ["examples", "ctrlc_handler"] }}
        //! polars = {{ version = "0.34", default_features=false, features=["csv", "dtype-i8", "dtype-i16", "dtype-u8", "dtype-u16"] }}
        //! rand = {{version="0.8", features=["serde1"]}}
        //! rand_xoshiro = {{ version="0.6", features = ["serde1"] }}
        //! ```
        fn main() {{
        {code}
        }}
        """

            enumerate_string_lines(rust_script_contents)

            rust_script_file = tempfile.NamedTemporaryFile(mode="w", dir="./lace")
            rust_script_file.write(rust_script_contents)
            rust_script_file.flush()

            code_result = subprocess.run(["rust-script", rust_script_file.name])
            return code_result.returncode
        else:
            return 1


def execute_single_file(args):
    return process_file(args.file, args.language, args.version)


def process_dir(args):
    excluded_files = []
    if args.exclusion_file:
        for line in args.exclusion_file:
            line = line.rstrip()
            excluded_files.append(line)

    failures = 0
    for root, dirs, files in os.walk(args.dir):
        for file in files:
            if not file.endswith(".md"):
                continue

            file_path = os.path.join(root, file)

            if file_path in excluded_files:
                continue

            failures+=process_file(file_path, args.language, args.version)

    return failures


parser = argparse.ArgumentParser(
    description="Run all the code for a given language in an MD file"
)
subparsers = parser.add_subparsers(help="Must give a subcommand")

single_file_command=subparsers.add_parser("single-file", help="Test single file")
single_file_command.add_argument('language')
single_file_command.add_argument('file')
single_file_command.add_argument("version") # Version isn't necessary for python, so this could be improved
single_file_command.set_defaults(func=execute_single_file)

directory_command = subparsers.add_parser(
    "directory", help="Test on all MD files in a directory"
)
directory_command.add_argument("language")
directory_command.add_argument("dir")
directory_command.add_argument("version")
directory_command.add_argument("--exclusion-file", type=argparse.FileType())
directory_command.set_defaults(func=process_dir)

args = parser.parse_args()
retval = args.func(args)
exit(retval)
