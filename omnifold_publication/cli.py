"""Command-line interface for OmniFold publication packages."""

from __future__ import annotations

import argparse
from pathlib import Path

from .reader import list_systematics, load_events, load_metadata
from .validation import validate_package


def _print_metadata_summary(path: Path, include_columns: bool = False) -> None:
    metadata = load_metadata(path, enforce_version=True)
    publication = metadata.get("publication", {})
    observables = metadata.get("observables", [])

    print(f"Package: {path}")
    print(f"Format version: {metadata.get('format_version')}")
    print(f"Events: {publication.get('event_count', 'unknown')}")
    print(f"Observables: {len(observables)}")
    print(f"Systematics: {', '.join(list_systematics(metadata)) or 'none'}")

    if include_columns:
        df = load_events(path)
        print(f"Columns: {len(df.columns)}")
        for column in df.columns:
            print(f"  - {column}")


def inspect_command(args: argparse.Namespace) -> int:
    _print_metadata_summary(Path(args.path), include_columns=True)
    return 0


def summary_command(args: argparse.Namespace) -> int:
    _print_metadata_summary(Path(args.path), include_columns=False)
    return 0


def validate_command(args: argparse.Namespace) -> int:
    errors = validate_package(args.path)
    if errors:
        print("Validation failed:")
        for error in errors:
            print(f"  - {error}")
        return 1

    print("Validation passed.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="omnifold-publication",
        description="Inspect and validate OmniFold publication packages.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser(
        "inspect",
        help="show package metadata and columns",
    )
    inspect_parser.add_argument("path", help="package directory")
    inspect_parser.set_defaults(func=inspect_command)

    validate_parser = subparsers.add_parser("validate", help="validate a package")
    validate_parser.add_argument("path", help="package directory")
    validate_parser.set_defaults(func=validate_command)

    summary_parser = subparsers.add_parser(
        "summary",
        help="show a compact package summary",
    )
    summary_parser.add_argument("path", help="package directory")
    summary_parser.set_defaults(func=summary_command)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
