"""Backward-compatible alias module for positional_encodings.

Historically the file name was misspelled as `positional_encodigs.py` and many
test modules import it directly. Re-export the correct implementation from
`positional_encodings.py` to preserve compatibility.
"""

from .positional_encodings import *  # noqa: F401,F403
