import argparse
import os
import sys
from typing import Dict, List, Optional, Text, TYPE_CHECKING, Union

from cli import SubParsersAction

if TYPE_CHECKING:
    from pathlib import Path

def add_subparser(
    subparsers: SubParsersAction, parents: List[argparse.ArgumentParser]
) -> None:
    """Add all training parsers.

    Args:
        subparsers: subparser we are going to attach to
        parents: Parent parsers, needed to ensure tree structure in argparse
    """
    train_parser = subparsers.add_parser(
        "train",
        help="Trains a Tileai model using your NLU data.",
        parents=parents,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    add_config_param(train_parser)
    add_out_param(train_parser, help_text="Path form models")

    
    train_parser.set_defaults(func=run_training)

    
    
def add_config_param(
    parser: Union[argparse.ArgumentParser, argparse._ActionsContainer]) -> None:
    """Specifies path to training data."""
    parser.add_argument(
        "-f","--file",
        default="nlu.json",
        help="Paths to the NLU config file.",
    )

def add_out_param(
    parser: Union[argparse.ArgumentParser, argparse._ActionsContainer],
    help_text: Text,
    default: Optional[Text] = "models",
    required: bool = False,) -> None:
    parser.add_argument(
        "-o","--out",
        type=str,
        default=default,
        help=help_text,
        required=required and default is None,
    )

def run_training(args: argparse.Namespace, can_exit: bool = False) -> Optional[Text]:
    """Trains a model.

    Args:
        args: Namespace arguments.
        can_exit: If `True`, the operation can send `sys.exit` in the case
            training was not successful.

    Returns:
        Path to a trained model or `None` if training was not successful.
    """
    
    #if training_result.code != 0 and can_exit:
    #    sys.exit(training_result.code)
    
    from tileai import train
    import cli.utils
    nlu = cli.utils.get_validated_path(
        args.file, "file", "domain/nlu.json", none_is_valid=True
    )
    out = args.out
   
    result = train(nlu, out)
    return result.model