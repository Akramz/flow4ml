"""Command line interface for the flow package."""

import argparse
import sys


def main(args=None):
    """Run the main entrypoint for the flow package.

    Args:
        args: Command line arguments. If None, sys.argv is used.

    Returns:
        int: Return code.
    """
    parser = argparse.ArgumentParser(description="Flow CLI tool")
    parser.add_argument("--version", action="store_true", help="Print version and exit")

    args = parser.parse_args(args)

    if args.version:
        from flow import __version__

        print(f"Flow version: {__version__}")
        return 0

    # Add your command-line functionality here
    print("Welcome to Flow! Add your CLI commands here.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
