import sys
from src.loader import load_prompt
from src.logger import setup_logging, get_logger



def main():
    setup_logging("baba")
    logger = get_logger(__name__)
    prompt = load_prompt(sys.argv[1])

    logger.info(repr(prompt))
    


if __name__ == "__main__":
    main()