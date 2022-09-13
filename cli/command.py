import sys
import os
import argparse
import logging
from . import _program
from clint.textui import puts, indent, colored
from cli import train, query
from shared.exceptions import TileaiException

logger = logging.getLogger(__name__)


def create_argument_parser() -> argparse.ArgumentParser:
    """Parse all the command line arguments """

    parser = argparse.ArgumentParser(
        prog = _program,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Aggiungo una descrizione di tileai",
    )

    parser.add_argument(
        "--version",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Print installed tileai version",
    )

    parent_parser = argparse.ArgumentParser(add_help=False)
    #add_logging_options(parent_parser)
    parent_parsers = [parent_parser]

    subparsers = parser.add_subparsers(help="Tileai commands")

    
    #run.add_subparser(subparsers, parents=parent_parsers)
    #shell.add_subparser(subparsers, parents=parent_parsers)
    train.add_subparser(subparsers, parents=parent_parsers)
    query.add_subparser(subparsers, parents=parent_parsers)
    #interactive.add_subparser(subparsers, parents=parent_parsers)

    return parser

def main(args = sys.argv[1:]):
    arg_parser = create_argument_parser()
    cmdline_arguments = arg_parser.parse_args()
    sys.path.insert(1, os.getcwd())
    try:
        if hasattr(cmdline_arguments, "func"):
            cmdline_arguments.func(cmdline_arguments)
        #elif hasattr(cmdline_arguments, "version"):
        #    print_version()
        else:
            # user has not provided a subcommand, let's print the help
            logger.error("No command specified.")
            arg_parser.print_help()
            sys.exit(1)
    except TileaiException as e:
        # these are exceptions we expect to happen (e.g. invalid training data format)
        # it doesn't make sense to print a stacktrace for these if we are not in
        # debug mode
        logger.debug("Failed to run CLI command due to an exception.", exc_info=e)
        print_error(f"{e.__class__.__name__}: {e}")
        sys.exit(1)
    

if __name__ == '__main__':
    main()
