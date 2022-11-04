import argparse

parser = argparse.ArgumentParser(description='train script argparse')
parser.add_argument('--finetune',type=bool,default=True)
parser.add_argument('--generate_model',type=str,default="Pegasus-large")
parser.add_argument('--')