"""
Continuous runner — wraps worker.main() in an infinite loop.

Usage:
    python run_bot.py

Keep the terminal open (Windows) or detach with nohup / a process manager
on Mac/Linux/VPS:
    nohup python run_bot.py > bot.log 2>&1 &
"""

from __future__ import annotations

import logging
import sys
import time
import traceback
from datetime import datetime
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
load_dotenv()

INTERVAL_SECONDS = 300  # 5 minutes

ET = ZoneInfo("America/New_York")

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s ET  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logging.Formatter.converter = lambda *_: datetime.now(ET).timetuple()
log = logging.getLogger("run_bot")

import worker


def run_continuous_bot() -> None:
    log.info("Continuous BTC prediction bot started (interval=%ds)", INTERVAL_SECONDS)

    while True:
        try:
            worker.main()
        except Exception:
            log.error("Unhandled exception in worker.main():\n%s", traceback.format_exc())

        log.info("Sleeping %d seconds until next run...", INTERVAL_SECONDS)
        time.sleep(INTERVAL_SECONDS)


if __name__ == "__main__":
    run_continuous_bot()
