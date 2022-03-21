import os
import argparse
from model_mobilenet_v2 import load_model, test_model

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default=None, required = True, help="model's name to test")
parser.add_argument('--testset', type=str, default=None, required = True, help="dataset folder name")
args = parser.parse_args()

DATA_DIR = 'data/test'

#모델 로딩
model = load_model('model', args.model)
test_model(os.path.join(DATA_DIR, args.testset), model)