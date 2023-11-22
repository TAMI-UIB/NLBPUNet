
from src.base import ParseKwargs


class DataParser(ParseKwargs):
    CHOICES = {
    }


class ModelParser(ParseKwargs):
    CHOICES = {
        'hyper_channels': 4,
        'multi_channels': 1,
        'iter_stages': 3,
        'sampling': 4,
        'device': 'cpu',
    }


class LossParser(ParseKwargs):
    CHOICES = {
        "alpha": 0.1
    }


class OptimParser(ParseKwargs):
    CHOICES = {
        "lr": 1e-5,
        "scheduler": 'ReduceLROnPlateau',
    }


class TrainParser(ParseKwargs):
    CHOICES = {
        "max_epochs": 1000,
        "batch_size": 1,
        "metric_track_key": 'psnr',
        "metric_track_mode": "min"
    }
