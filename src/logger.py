import os, yaml, uuid
from logging import Filter, Logger, getLogger, basicConfig 
from logging.config import dictConfig
from pathlib import Path


class RunFilter(Filter):
    def __init__(self, delimiter, length, additional_fields=None):
        self.delimiter = delimiter
        self.length = length
        self.execution_id = str(uuid.uuid4())
        self.additional_fields = additional_fields if additional_fields else {}
        super().__init__()

    def filter(self, record):
        record.execution_id = self.execution_id
        return True

    def emit_delimiter(self, logger):
        delimiter_message = (
            f'{self.delimiter * self.length} Execution Start | '
            f'Execution ID: {self.execution_id} | ' + 
            ''.join(f"{str(k).capitalize()}: {str(v)}" for k, v in self.additional_fields.items())
            + f' {self.delimiter * self.length}'
        )
        logger.debug("")
        logger.debug(delimiter_message)

def setup_logging(configfile: str|Path="", **run_args):

    if isinstance(configfile, str):
        configfile = Path(configfile)

    """Setup logging configuration"""
    if configfile is not None and configfile.is_file():
        with configfile.open("r") as f:
            config = yaml.safe_load(f)

        for a, v in run_args.items():
            config["filters"]["run"]["additional_fields"][a] = v
        
        dictConfig(config)
        logger = getLogger()
        for handler in logger.handlers:
            if (filter_instance := next(iter(f for f in handler.filters if isinstance(f, RunFilter)), None)) is None:
                continue

            logger.debug(f"Filter '{filter_instance}' found for handler '{handler}'")
            filter_instance.emit_delimiter(logger)
            break
    else:
        default_level = os.getenv("LOG_LEVEL")
        basicConfig(level=default_level)
        logger = get_logger()
        logger.warning(f"Warning: Logging configuration file not found at {configfile}. Using default level: {default_level}")


def get_logger(_: str=None) -> Logger:
    return getLogger()


