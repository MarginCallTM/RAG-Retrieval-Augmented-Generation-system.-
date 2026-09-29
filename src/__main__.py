"""Command-line entry point: ``uv run python -m src <command> [options]``."""

import sys

import fire

from src.cli import CliError, RagCli


def main() -> None:
    """Dispatch the command line to ``RagCli``, without tracebacks.

    User errors exit with code 1 and a one-line message. Ctrl+C exits with
    code 130, the shell convention for an interrupted program.
    """
    try:
        fire.Fire(RagCli)
    except CliError as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()
