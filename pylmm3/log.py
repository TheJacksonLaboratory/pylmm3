"""Logging configuration for pylmm3.

pylmm3 is a library.  It never installs handlers or calls basicConfig — that
is the application's job (pylmmGWAS, pylmmKinship, or any orchestrator such as
plinkformatter that calls pylmm3 in-process).

At import time a NullHandler is attached to the "pylmm3" logger so that if no
application configures logging, pylmm3 is completely silent (the stdlib-
recommended pattern for libraries; see Python docs "Logging HOWTO").

configure() is provided for application entry points.  It sets the level on
the "pylmm3" logger (and therefore all pylmm3.* children) — nothing more.
Handler installation and formatting are the caller's responsibility.

Level precedence when called without an explicit argument:
  1. PYLMM3_LOG_LEVEL env var  (DEBUG | INFO | WARNING | ERROR)
  2. WARNING  (default — silent)
"""

import logging
import os
import sys

_ENV_VAR = "PYLMM3_LOG_LEVEL"
_FORMAT  = "[%(levelname)-7s] %(asctime)s.%(msecs)03d  %(name)s  %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"

# Standard library pattern: attach NullHandler so pylmm3 is silent unless the
# application configures a real handler.
logging.getLogger("pylmm3").addHandler(logging.NullHandler())


def env_level() -> int:
    """Return the log level from PYLMM3_LOG_LEVEL, defaulting to WARNING."""
    val = os.environ.get(_ENV_VAR, "").upper()
    return getattr(logging, val, logging.WARNING)


def configure(level: int | None = None) -> None:
    """Set the verbosity of all pylmm3.* loggers.

    Does NOT install a handler — call this when an application (e.g.
    plinkformatter) already owns the root handler and just needs to control
    how loud pylmm3 is.

    Args:
        level: explicit level (e.g. logging.DEBUG).  If None, reads
               PYLMM3_LOG_LEVEL; falls back to WARNING.
    """
    effective = level if level is not None else env_level()
    logging.getLogger("pylmm3").setLevel(effective)


def setup(level: int | None = None) -> None:
    """Install a root handler (if none exists) then set the pylmm3 log level.

    For use by pylmm3's own CLI entry points (pylmmGWAS, pylmmKinship) that
    are the application.  Orchestrators that already own a root handler (e.g.
    plinkformatter) should call configure() instead.

    Args:
        level: explicit level.  If None, reads PYLMM3_LOG_LEVEL; falls back
               to WARNING.
    """
    if not logging.root.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
        logging.root.addHandler(handler)
    configure(level)
