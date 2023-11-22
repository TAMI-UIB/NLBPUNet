import argparse

from dotenv import load_dotenv

from config import ModelParser, DataParser, LossParser, OptimParser, TrainParser
from src.base import Experiment
from src.dataset import dict_dataset
from src.loss import dict_loss
from src.model import dict_model
from src.utils.optimizers import dict_optimizer

load_dotenv()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add new parameters in config.py")
    parser.add_argument('method', choices=['train', 'test', 'classic_methods'])
    parser.add_argument("--dataset", type=str, help="Dataset name", default=list(dict_dataset.keys())[0], choices=dict_dataset.keys())
    parser.add_argument("--model", type=str, help="Model name", default=list(dict_model.keys())[0], choices=dict_model.keys())
    parser.add_argument("--loss", type=str, help="Loss name", default=list(dict_loss.keys())[0], choices=dict_loss.keys())
    parser.add_argument("--optimizer", type=str, help="Optimizer name", default=list(dict_optimizer.keys())[0], choices=dict_optimizer.keys())

    parser.add_argument("--data-params", nargs='*', action=DataParser, default=DataParser.CHOICES, help="Dataset parameters")
    parser.add_argument("--model-params", nargs='*', action=ModelParser, default=ModelParser.CHOICES, help="Model parameters")
    parser.add_argument("--loss-params", nargs='*', action=LossParser, default=LossParser.CHOICES, help="Loss parameters")
    parser.add_argument("--optim-params", nargs='*', action=OptimParser, default=OptimParser.CHOICES, help="Optim parameters")
    parser.add_argument("--train-params", nargs='*', action=TrainParser, default=TrainParser.CHOICES, help="Train parameters")

    parser.add_argument("--resume-path", type=str, help="Resume path")
    parser.add_argument("--output-path", type=str, default="runs/", help="Output path")

    args = parser.parse_args()
    exp = Experiment(**args.__dict__)
    if args.method == 'train':
        exp.train()
    elif args.method == 'test':
        exp.test(args.output_path)
    elif args.method == 'classic_methods':
        exp.classical_methods(args.output_path)
