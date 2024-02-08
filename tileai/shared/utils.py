import asyncio
import functools
import importlib
import inspect
import logging
import shutil
import os
from typing import Text, Dict, Optional, Any, List, Callable, Collection, Type


def minimal_kwargs(
    kwargs: Dict[Text, Any], func: Callable, excluded_keys: Optional[List] = None
) -> Dict[Text, Any]:
    """Returns only the kwargs which are required by a function. Keys, contained in
    the exception list, are not included.

    Args:
        kwargs: All available kwargs.
        func: The function which should be called.
        excluded_keys: Keys to exclude from the result.

    Returns:
        Subset of kwargs which are accepted by `func`.

    """

    excluded_keys = excluded_keys or []

    possible_arguments = arguments_of(func)

    return {
        k: v
        for k, v in kwargs.items()
        if k in possible_arguments and k not in excluded_keys
    }

def arguments_of(func: Callable) -> List[Text]:
    """Return the parameters of the function `func` as a list of names."""
    import inspect

    return list(inspect.signature(func).parameters.keys())

import shutil
import os

def copy_directory(src_dir, dst_dir, dirs_exist_ok=True, ignore_patterns=[]):
    """Copies a directory and its contents recursively to a new directory.

    Args:
        src_dir (str): Path to the source directory.
        dst_dir (str): Path to the destination directory.
        dirs_exist_ok (bool, optional): If True, will silently create the destination
            directory if it doesn't exist. Defaults to True.
        ignore_patterns (list[str], optional): A list of file and directory patterns
            to exclude from copying. Defaults to an empty list.

    Raises:
        FileNotFoundError: If the source directory does not exist.
        OSError: If there are issues with copying files or directories.
    """

    if not os.path.exists(src_dir):
        raise FileNotFoundError(f"Source directory '{src_dir}' not found.")

    # Create the destination directory if it doesn't exist
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir, exist_ok=dirs_exist_ok)

    # Helper function to check if a file/directory should be ignored
    def should_ignore(path):
        for pattern in ignore_patterns:
            if fnmatch.fnmatch(path, pattern):
                return True
        return False

    # Recursively copy files and directories
    for dirpath, dirnames, filenames in os.walk(src_dir):
        # Exclude ignored directories
        dirnames[:] = [d for d in dirnames if not should_ignore(os.path.join(dirpath, d))]

        # Exclude ignored files
        filenames[:] = [f for f in filenames if not should_ignore(os.path.join(dirpath, f))]

        # Create destination subdirectory if needed
        dest_dir = os.path.join(dst_dir, os.path.relpath(dirpath, src_dir))
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir, exist_ok=dirs_exist_ok)

        # Copy files
        for filename in filenames:
            src_file = os.path.join(dirpath, filename)
            dst_file = os.path.join(dest_dir, filename)
            try:
                shutil.copy2(src_file, dst_file)  # Preserve file metadata
            except OSError as e:
                print(f"Error copying file '{src_file}': {e}")

