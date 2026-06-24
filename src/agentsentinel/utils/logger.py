import logging

RESET  = "\033[0m"
BOLD   = "\033[1m"

COLORS = {
    logging.DEBUG:    "\033[36m",   # cyan
    logging.INFO:     "\033[32m",   # green
    logging.WARNING:  "\033[33m",   # yellow
    logging.ERROR:    "\033[31m",   # red
    logging.CRITICAL: "\033[35m",   # magenta
}


class SentinelFormatter(logging.Formatter):
    PREFIX = "AgentSentinel"

    def format(self, record: logging.LogRecord) -> str:
        color = COLORS.get(record.levelno, RESET)
        level = record.levelname
        msg   = super().format(record)
        # Strip the logger name from the message — replace with PREFIX
        short = msg.split(" - ", 2)[-1]
        return f"{color}{BOLD}[{self.PREFIX}]{RESET} {color}{level}{RESET} — {short}"


def setup_logger(name: str = "agentsentinel", level: int = logging.DEBUG) -> None:
    logger = logging.getLogger(name)
    # Strip any prior handlers so setup_logger is idempotent and resilient to
    # anything an earlier caller or import may have attached.
    logger.handlers.clear()
    handler = logging.StreamHandler()
    handler.setFormatter(SentinelFormatter())
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
