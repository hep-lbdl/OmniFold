"""Custom exceptions for the OmniFold publication package."""


class OmniFoldPublicationError(Exception):
    """Base exception for all omnifold_publication errors."""


class PackageWriteError(OmniFoldPublicationError):
    """Raised when writing a publication package fails."""


class PackageReadError(OmniFoldPublicationError):
    """Raised when reading a publication package fails."""


class PackageValidationError(OmniFoldPublicationError):
    """Raised when a publication package fails validation."""


class UnsupportedFormatVersion(OmniFoldPublicationError):
    """Raised when a package uses an unsupported format_version."""
