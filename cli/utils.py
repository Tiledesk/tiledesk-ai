import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional, TYPE_CHECKING, Text, Union, overload
import warnings


if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

def get_validated_path(
    current: Optional[Union["Path", Text]],
    parameter: Text,
    default: Optional[Union["Path", Text]] = None,
    none_is_valid: bool = False,
) -> Optional[Union["Path", Text]]:
    """Checks whether a file path or its default value is valid and returns it.

    Args:
        current: The parsed value.
        parameter: The name of the parameter.
        default: The default value of the parameter.
        none_is_valid: `True` if `None` is valid value for the path,
                        else `False``

    Returns:
        The current value if it was valid, else the default value of the
        argument if it is valid, else `None`.
    """
    if current is None or current is not None and not os.path.exists(current):
        if default is not None and os.path.exists(default):
            reason_str = f"'{current}' not found."
            if current is None:
                reason_str = f"Parameter '{parameter}' not set."
            else:
                warnings.warn(
                    f"The path '{current}' does not seem to exist. Using the "
                    f"default value '{default}' instead."
                )

            logger.debug(f"{reason_str} Using default location '{default}' instead.")
            current = default
        elif none_is_valid:
            current = None
        else:
            cancel_cause_not_found(current, parameter, default)

    return current
