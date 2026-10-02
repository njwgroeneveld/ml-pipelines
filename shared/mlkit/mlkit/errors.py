"""Exit codes shared by every pipeline step.

Argo retries a failed step unless it exits with EXIT_DATA_CHECK: retrying
does not fix data that is missing or wrong.
"""
import sys
from collections.abc import Callable

EXIT_DATA_CHECK = 2


class DataCheckError(Exception):
    """The input data is unusable; the step must fail without a retry."""


def run_main(main: Callable[[], None]) -> None:
    """Run a step's main(); turn a DataCheckError into exit code 2."""
    try:
        main()
    except DataCheckError as exc:
        print(f"DATA CHECK FAILED: {exc}", file=sys.stderr)
        sys.exit(EXIT_DATA_CHECK)
