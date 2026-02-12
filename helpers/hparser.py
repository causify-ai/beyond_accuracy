"""
Import as:

import helpers.hparser as hparser
"""

import argparse
import logging
from typing import Optional

_LOG = logging.getLogger(__name__)


def add_verbosity_arg(
    parser: argparse.ArgumentParser, *, log_level: str = "INFO"
) -> argparse.ArgumentParser:
    parser.add_argument(
        "-v",
        dest="log_level",
        default=log_level,
        # TRACE=5
        # DEBUG=10
        # INFO=20
        # WARNING=30
        # CRITICAL=50
        choices=["TRACE", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level",
    )
    return parser


# #############################################################################
# Command line options for parallel processing.
# #############################################################################


# pylint: disable=line-too-long
# TODO(gp): These should go in hjoblib.py
def add_parallel_processing_arg(
    parser: argparse.ArgumentParser,
    *,
    num_threads_default: Optional[str] = None,
) -> argparse.ArgumentParser:
    """
    Add parallel processing args.

    The "incremental idiom" means skipping processing computation that has
    already been performed. E.g., if we need to transform files from one dir to
    another we skip the files already processed (assuming that a file present
    in the destination dir is an indication that it has already been
    processed).

    The default behavior should always be incremental since "incremental mode"
    is not destructive like the non-incremental, i.e., delete and restart

    The incremental behavior  is disabled with `--no_incremental`. This implies
    performing the computation in any case
    - It is often implemented by deleting the destination dir and then running
      again, even in incremental mode
    - If the destination dir already exists, then we require the user to
      explicitly use `--force` to confirm that the user knows what is doing
    """
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print the workload and exit without running it",
    )
    parser.add_argument(
        "--no_incremental",
        action="store_true",
        help="Skip workload already performed",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Confirm that one wants to remove the previous results. It works only together with --no_incremental",
    )
    #
    help = """
    Number of threads to use:
    - '-1' to use all CPUs;
    - '1' to use one-thread at the time but using the parallel execution (mainly used
    for debugging)
    - 'serial' to serialize the execution without using parallel execution"""
    if num_threads_default is None:
        parser.add_argument(
            "--num_threads",
            action="store",
            help=help,
            required=True,
        )
    else:
        parser.add_argument(
            "--num_threads",
            action="store",
            help=help,
            default=num_threads_default,
        )
    parser.add_argument("--no_keep_order", action="store_true", help="")
    parser.add_argument(
        "--num_func_per_task",
        action="store",
        type=int,
        default=None,
        help="Number of function execute in a (parallel) task of the workload. `None` means automatically decided by the function",
    )
    parser.add_argument(
        "--skip_on_error",
        action="store_true",
        help="Continue execution after encountering an error",
    )
    parser.add_argument(
        "--num_attempts",
        default=1,
        type=int,
        help="Repeat running an experiment up to `num_attempts` times",
        required=False,
    )
    return parser