"""
Authors: wucunyin
"""

from utils.configs import *
from utils.utils import *



def main():
    parser = define_args() # args in utils.py

    args = parser.parse_args()
    config(args)

    ddp_init(args)

    runner = Runner(args)
    
    if not args.evaluate:
        runner.train()
    else:
        runner.eval()


if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    main()

