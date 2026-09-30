"""Console + rotating file logging."""

from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path


def setup_logging(level: str = "INFO", log_dir: Path | None = None) -> None:
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    for handler in list(root.handlers):
        root.removeHandler(handler)
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%H:%M:%S")
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    root.addHandler(console)
    if log_dir is not None:
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
            fh = logging.handlers.RotatingFileHandler(
                log_dir / "literature_buddy.log", maxBytes=2_000_000, backupCount=3, encoding="utf-8"
            )
            fh.setFormatter(fmt)
            root.addHandler(fh)
        except OSError:
            root.warning("Could not create log file in %s", log_dir)
    for noisy in ("httpx", "httpcore", "urllib3", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
