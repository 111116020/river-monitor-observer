import asyncio
import logging


class CustomFormatter(logging.Formatter):

    _grey = "\033[1;30m"
    _purple = "\033[35m"
    _green = "\033[1;32m"
    _blue = "\033[1;34m"
    _yellow = "\033[1;33m"
    _red = "\033[1;31m"
    _reset = "\033[0m"

    _format = f"{_grey}%(asctime)s{_reset} [{{level_color}}%(levelname).4s{_reset}]\t{_purple}%(name)s{_reset} %(message)s"

    FORMATS = {
        logging.DEBUG:      _format.format(level_color=_green),
        logging.INFO:       _format.format(level_color=_blue),
        logging.WARNING:    _format.format(level_color=_yellow),
        logging.ERROR:      _format.format(level_color=_red),
        logging.CRITICAL:   _format.format(level_color=_red),
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


async def exec_periodically(interval_sec: int, coro_name, *args, **kwargs):
    while True:
        await coro_name(*args, **kwargs)
        await asyncio.sleep(interval_sec)
