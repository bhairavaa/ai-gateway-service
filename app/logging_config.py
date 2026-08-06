"""Logging configuration — the one place that configures the stdlib `logging`
module for the whole process, called once at startup. Plain stdlib logging with
a formatter is enough for a single-process gateway; not worth a structured-
logging dependency at this scale.
"""

import logging

from app.config import Settings


def configure_logging(settings: Settings) -> None:
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )
