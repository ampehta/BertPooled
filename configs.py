import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--max_len',default=128)
parser.add_argument('--batch_size',default=32)
parser.add_argument('--epochs',default=10)
parser.add_argument('--learning_rate',default=1e-3)
parser.add_argument('--per_iter',default=50)

args=parser.parse_args()