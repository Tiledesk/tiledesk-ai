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
    """Add query parsers.

    Args:
        subparsers: subparser we are going to attach to
        parents: Parent parsers, needed to ensure tree structure in argparse
    """
    train_parser = subparsers.add_parser(
        "query",
        help="Query a Tileai model using.",
        parents=parents,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    add_config_param(train_parser)
    add_text_param(train_parser, help_text="Your Text")

    
    train_parser.set_defaults(func=run_query)

    
    
def add_config_param(
    parser: Union[argparse.ArgumentParser, argparse._ActionsContainer]) -> None:
    """Specifies path to model."""
    parser.add_argument(
        "-m","--model",
        default="default.pt",
        help="Paths to the NLU model file.",
    )

def add_text_param(
    parser: Union[argparse.ArgumentParser, argparse._ActionsContainer],
    help_text: Text) -> None:
    parser.add_argument(
        "-t","--text",
        type=str,
        help=help_text,
    )

def run_query(args: argparse.Namespace, can_exit: bool = False) -> Optional[Text]:
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
    try:
        from tileai import query
        import cli.utils
        model = cli.utils.get_validated_path(
            args.model, "model", "models/default", none_is_valid=True
        )
        query_text = args.text
        
        label, risult_dict = query(model, query_text)
        print ("label: ",risult_dict["intent"]["name"], "- confidence: ", risult_dict["intent"]["confidence"])
        print ("=================================")
        print ("intent ranking")
        for elem in risult_dict["intent_ranking"]:
            print("\tlabel: ",elem["name"], "- confidence: ", elem["confidence"])

    except Exception as e:
            print(e)
    