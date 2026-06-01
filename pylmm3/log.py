"""Logging configuration for pylmm3.

Library callers should configure their own root handler; pylmm3 modules only 
call logging.getLogger(__name__) and never touch handlers.

configure() is called by the CLI entry points (pylmmGWAS, pylmmKinship).  It:
  - sets the "pylmm3" package logger to the requested level (so all pylmm3.*
    loggers inherit it),
  - installs a fallback StreamHandler on the root logger only when no handler
    exists yet (i.e. the library was not imported into an already-configured
    logging environment).

Level precedence (highest wins):
  1. CLI flag  --log-level / --verbose
  2. Env var   PYLMM3_LOG_LEVEL=DEBUG|INFO|WARNING|ERROR
  3. Default   WARNING
"""

import logging
import os
import sys

_FORMAT  = "[%(levelname)-7s] %(asctime)s.%(msecs)03d  %(name)s  %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"
_ENV_VAR = "PYLMM3_LOG_LEVEL"


def env_level() -> int:
    """Return the log level from PYLMM3_LOG_LEVEL, defaulting to WARNING."""
    val = os.environ.get(_ENV_VAR, "").upper()
    return getattr(logging, val, logging.WARNING)


def configure(level: int | None = None) -> None:
    """Configure pylmm3 package logging.

    Args:
        level: explicit level (e.g. logging.DEBUG).  If None, reads
               PYLMM3_LOG_LEVEL env var; falls back to WARNING.
    """
    effective = level if level is not None else env_level()
    logging.getLogger("pylmm3").setLevel(effective)
    if not logging.root.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
        logging.root.addHandler(handler)
