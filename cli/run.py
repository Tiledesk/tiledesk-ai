import argparse
import logging
import os
from typing import List, Text, Union,Optional

from cli import SubParsersAction


logger = logging.getLogger(__name__)


def add_subparser(
    subparsers: SubParsersAction, parents: List[argparse.ArgumentParser]
) -> None:
    """Add all run parsers.

    Args:
        subparsers: subparser we are going to attach to
        parents: Parent parsers, needed to ensure tree structure in argparse
    """
    run_parser = subparsers.add_parser(
        "run",
        parents=parents,
        conflict_handler="resolve",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Starts a Tileai server.",
    )
    add_port_param(run_parser)
    run_parser.set_defaults(func=run_server)

   

   
def add_port_param(
    parser: Union[argparse.ArgumentParser, argparse._ActionsContainer]) -> None:
    """Specifies port param."""
    parser.add_argument(
        "-p","--port",
        default="5100",
        help="Port for http server.",
    )

def run_server(args: argparse.Namespace, can_exit: bool = False) -> Optional[Text]:
    from tileai import run

    print(args.port) 
    run(args.port)
   