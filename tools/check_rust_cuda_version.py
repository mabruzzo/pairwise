#!/usr/bin/env python3
"""
A simple script designed to look through Cargo.toml files for a set of dependencies
from the Rust-CUDA project. If any one of the crates are linked to the GitHub
repository, this script ensures they are all consistent.

I'm pretty sure that this can't be done directly by Cargo since we need to compile
CUDA kernels using a nightly toolchain while we try to use to stable in all other
cases. There's a good chance that we are being more strict than necessary, but it's
better to be safe than sorry.
"""

import argparse
from collections import defaultdict
from dataclasses import dataclass
import pathlib
import os
import sys
from typing import Dict, Iterable, Iterator, List, Optional, Set

if sys.version_info < (3, 11):
    import tomli as tomllib
else:
    import tomllib


def find_toml_files(root_path: os.PathLike) -> Iterator[pathlib.Path]:
    for dirpath, dirname, filenames in os.walk(root_path, topdown=True):
        for name in filenames:
            if name.lower().endswith(".toml"):
                yield pathlib.Path(dirpath, name)


@dataclass
class GitLocation:
    # details for specifying a package location with git
    git: Optional[str]
    tag: Optional[str]
    rev: Optional[str]


@dataclass
class DependencySpecificationProps:
    path: os.PathLike  # file containing the specification
    table_name: str  # the name of the toml table that contained the entry
    git_props: GitLocation


def collect_info_about_specified_dependencies(
    dependency_names: Set[str], tomlfile_paths: Iterable[os.PathLike]
) -> Dict[str, List[DependencySpecificationProps]]:
    """Returns a dict describing all occurrences of the specified dependencies in
    Cargo.toml files
    """

    def gather_info(out_dict, tomlpath, conf_info, table_name):
        for depname, value in conf_info.get(table_name, {}).items():
            if depname not in dependency_names:
                continue

            value = {"version": value} if isinstance(value, str) else value

            git_props = None
            if "git" in value:
                git_props = GitLocation(
                    git=value["git"], tag=value.get("tag"), rev=value.get("rev")
                )
            # print(out_dict)
            out_dict[depname].append(
                DependencySpecificationProps(tomlpath, table_name, git_props)
            )

    out_dict = defaultdict(list)

    for tomlpath in iter(tomlfile_paths):
        if str(tomlpath.name).lower() != "cargo.toml":
            continue
        with open(tomlpath, "rb") as f:
            conf_info = tomllib.load(f)
            gather_info(out_dict, tomlpath, conf_info, "dependencies")
            gather_info(out_dict, tomlpath, conf_info, "dev-dependencies")
            gather_info(out_dict, tomlpath, conf_info, "build-dependencies")
    return dict(out_dict)


_descr, _epilog = __doc__.split("\n\n")

_parser = argparse.ArgumentParser(description=_descr, epilog=_epilog)
_parser.add_argument("--show-all-dep-occurrences", action="store_true")
_parser.add_argument(
    "--root-path", help="path to the root of the repository", required=False
)


def main(**parse_args_kwargs):
    args = _parser.parse_args(**parse_args_kwargs)

    # let's go through and record the dependency information

    if args.root_path is not None:
        root_path = args.root_path
    else:
        current_file_location = pathlib.Path(__file__)
        assert current_file_location.is_file()  # sanity check!
        tool_dir = current_file_location.parent
        root_path = tool_dir.parent

    if not pathlib.Path(root_path, ".git").is_dir():
        raise RuntimeError("can't find a .git file in root_path")

    # these packages are all inter-related
    # (it would be nice if they weren't hardcoded)
    known_packages = set(
        [
            "cust",
            "cust_raw",
            "cust_core",
            "cuda_builder",
            "cuda_std",
        ]
    )

    itr = find_toml_files(root_path)

    occurrences = collect_info_about_specified_dependencies(known_packages, itr)

    if len(occurrences) == 0:
        raise RuntimeError(
            "can't find dependencies on any of the following packages: "
            f"{', '.join(repr(pkg) for pkg in known_packages)}",
            file=sys.stderr,
        )
        return 1

    if args.show_all_dep_occurrences:
        print("Occurrences of the Rust-CUDA deps:")
        for k, cases in occurrences.items():
            print(f"  {k}:")
            for case in cases:
                print(f"  -> in the {case.table_name!r} table of {case.path!s}")
        return 0

    else:
        first_key = next(iter(occurrences))
        ref_case = occurrences[first_key][0]
        ref_git_props = ref_case.git_props

        for key, cases in occurrences.items():
            for case in cases:
                if case.git_props != ref_git_props:
                    print(
                        f"""\
There is an inconsistency in the location that Cargo should fetching
the Rust-CUDA packages from. If one dependency comes from Git, then
they should all come from git. There is an inconsistency between

-> the {first_key!r} package listed in the {ref_case.table_name!r} table
   of {ref_case.path!s}
-> the {key!r} package listed in the {case.table_name!r} table
   of {case.path!s}""",
                        file=sys.stderr,
                    )
                    return 1
        return 0


if __name__ == "__main__":
    sys.exit(main())
