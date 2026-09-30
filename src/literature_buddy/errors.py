"""Exception types shared across layers (kept tiny so every layer can import them)."""


class LiteratureBuddyError(Exception):
    """Base class; the message is safe to show to the user."""


class Cancelled(LiteratureBuddyError):
    """Raised inside long tasks when the user cancels."""


class ParseError(LiteratureBuddyError):
    pass


class DownloadError(LiteratureBuddyError):
    pass


class ModelUnavailableError(LiteratureBuddyError):
    """A model backend cannot be reached / loaded. The message says how to fix it."""
