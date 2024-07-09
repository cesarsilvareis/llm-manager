import sys
from src.loader import load_modelcfg_from_fs, load_prompt, load_executions
from src.logger import setup_logging
from src.inference import Inference
from src.evaluation import TruthfulQA

from argparse import ArgumentParser

class CustomArgumentParser(ArgumentParser):
    def error(self, message):
        sys.stderr.write(f'Error: {message}\n')
        self.print_help()
        sys.exit(2)
    
    def parse_args(self, args=None, namespace=None):
        args = super().parse_args(args, namespace)
        
        dependencies = [
            [   # Inference
                (args.prompt, "prompt"),
                (args.modelcfg, "modelcfg"),
                (args.resfile, "resfile")
            ],
            [   # Evaluation
                (args.bench, "bench"),
                (args.modelcfg, "modelcfg"),
                (args.resfile, "resfile")
            ]
        ]

        for dpd in dependencies:
            if len(dpd) <= 1 or dpd[0][0] is None:
                continue

            defined_args = set((arg, n) for arg, n in dpd if arg is not None)

            undefined_args = set(dpd) - defined_args
            if len(undefined_args) > 0:
                self.error(f'Arguments "{",".join(f"--{n}" for _, n in defined_args)}" requires "{",".join(f"--{n}" for _, n in undefined_args)}" to be set.')
    
        return args

def parse_arguments():
    parser = CustomArgumentParser(description="Tool for experiment LLMs in restricted, priviledged environments")
    parser.add_argument("--execfile", "-e", type=str, required=False)
    parser.add_argument("--modelcfg", "-c", type=str, required=False)
    parser.add_argument("--prompt", "-p", type=str, required=False)
    parser.add_argument("--bench", "-b", type=str, required=False)
    parser.add_argument("--resfile", "-o", type=str, required=False)
    parser.add_argument(
        "--log", "-l", type=str, required=False,
        help="Log configuration file" 
    )

    return parser.parse_args()

def main():
    args = parse_arguments()

    setup_logging(args.log, execfile=args.execfile)

    if args.prompt is not None:
        inf = Inference(id=-1, 
                modelcfg=load_modelcfg_from_fs(args.modelcfg), 
                prompt=load_prompt(args.prompt),
                output_filename=args.resfile
            )
        inf.run(single=True)

    if args.bench is not None:
        ben = TruthfulQA(id=-1,
            modelcfg=load_modelcfg_from_fs(args.modelcfg),
            
        )
        ben.run(single=True, explore_comb=False)
    
    if args.execfile is not None:
        executions = load_executions(args.execfile, args.resfile)
        for exec in executions:
            exec.run()

if __name__ == "__main__":
    main()