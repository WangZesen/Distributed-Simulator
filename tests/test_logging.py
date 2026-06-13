import re
from io import StringIO

from loguru import logger

from distributed_simulator.logging import LOG_FORMAT


def test_log_format_is_compact_and_includes_source_location() -> None:
    output = StringIO()
    sink_id = logger.add(output, format=LOG_FORMAT, colorize=False)
    try:
        logger.info("compact message")
    finally:
        logger.remove(sink_id)

    assert re.fullmatch(
        r"\d{2}:\d{2}:\d{2} \| test_logging\.py:\d+ \| compact message\n",
        output.getvalue(),
    )
