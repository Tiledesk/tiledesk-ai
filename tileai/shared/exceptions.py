import json
from typing import Optional, Text, Union, Any
from http import HTTPStatus
from tileai.cli import __version__

import jsonschema




class TileaiException(Exception):
    """Base exception class for all errors raised by tailai.

    These exceptions results from invalid use cases and will be reported
    to the users.
    """


class TileaiCoreException(TileaiException):
    """Basic exception for errors raised by Tileai."""


class InvalidParameterException(TileaiException, ValueError):
    """Raised when an invalid parameter is used."""


class FileNotFoundException(TileaiException, FileNotFoundError):
    """Raised when a file, expected to exist, doesn't exist."""


class FileIOException(TileaiException):
    """Raised if there is an error while doing file IO."""


class InvalidConfigException(ValueError, TileaiException):
    """Raised if an invalid configuration is encountered."""


class UnsupportedFeatureException(TileaiCoreException):
    """Raised if a requested feature is not supported."""


class SchemaValidationError(TileaiException, jsonschema.exceptions.ValidationError):
    """Raised if schema validation via `jsonschema` failed."""


class InvalidEntityFormatException(TileaiException, json.JSONDecodeError):
    """Raised if the format of an entity is invalid."""

    @classmethod
    def create_from(
        cls, other: json.JSONDecodeError, msg: Text
    ) -> "InvalidEntityFormatException":
        """Creates `InvalidEntityFormatException` from `JSONDecodeError`."""
        return cls(msg, other.doc, other.pos)


class ConnectionException(TileaiException):
    """Raised when a connection to a 3rd party service fails.

    It's used by our broker and tracker store classes, when
    they can't connect to services like postgres, dynamoDB, mongo.
    """

class ErrorResponse(Exception):
    """Common exception to handle failing API requests."""

    def __init__(
        self,
        status: Union[int, HTTPStatus],
        reason: Text,
        message: Text,
        details: Any = None,
    ) -> None:
        """Creates error.

        Args:
            status: The HTTP status code to return.
            reason: Short summary of the error.
            message: Detailed explanation of the error.
            details: Additional details which describe the error. Must be serializable.
        """
        self.error_info = {
            "version": __version__,
            "status": "error",
            "message": message,
            "reason": reason,
            "details": details or {},
            "code": status,
        }
        self.status = status
        
        super(ErrorResponse, self).__init__()
