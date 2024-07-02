from src.loader import load_prompt, load_executions
from src.logger import setup_logging, get_logger
from argparse import ArgumentParser

def parse_arguments():
    parser = ArgumentParser(description="Tool for experiment LLMs in restricted, priviledged environments")
    parser.add_argument("--execfile", "-e", type=str, required=True)
    parser.add_argument("--resfile", "-o", type=str, required=False)
    parser.add_argument(
        "--log", "-l", type=str, required=False,
        help="Log configuration file" 
    )

    args = parser.parse_args()

    return args

def main():
    args = parse_arguments()

    setup_logging(args.log, execfile=args.execfile)

    executions = load_executions(args.execfile, args.resfile)
    print(executions)

    for exec in executions:
        exec.run()

    

if __name__ == "__main__":
    main()