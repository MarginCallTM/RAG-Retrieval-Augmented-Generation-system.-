"""Command-line entry point: ''uv run python -m src <command> [options]''."""

import fire

from src.cli import RagCli


def main() -> None:
    """Dispatch the command line to the matching ''RagCli'' method"""
    fire.Fire(RagCli)


if __name__ == "__main__":
    main()
