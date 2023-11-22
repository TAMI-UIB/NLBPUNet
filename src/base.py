import argparse
import gc
import os
from datetime import datetime as dt

import numpy as np
import torch
from torch.nn.functional import fold, unfold
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import dict_dataset
from src.loss import dict_loss
from src.model import dict_model
from src.utils.constants import TRAIN, VAL, TEST
from src.utils.metrics import MetricCalculatorExample
from src.utils.optimizers import dict_optimizer_scheduler, dict_optimizer, SCHED_STEP_PARAM
from src.utils.visualization import TensorboardWriter, FileWriter

gc.collect()
torch.cuda.empty_cache()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Experiment:
    def __init__(self, dataset: str, model: str, loss: str, optimizer: str,
                 data_params: dict, model_params: dict, loss_params: dict, optim_params: dict,
                 train_params: dict, resume_path: str = None, **kwargs):
        self.experiment = {'dataset': dataset, 'model': model, 'loss': loss, 'optimizer': optimizer,
                           'data_params': data_params, 'model_params': model_params, 'loss_params': loss_params,
                           'optim_params': optim_params, 'train_params': train_params, 'resume_path': resume_path}

        self.dataset_name = dataset
        self.dataset = dict_dataset[dataset]
        self.data_params = data_params

        self.model_name = model
        self.model = dict_model[model]
        self.model_params = model_params

        self.optimizer = dict_optimizer[optimizer]
        self.optim_params = optim_params
        self.scheduler_name = optim_params.pop('scheduler', list(dict_optimizer_scheduler.keys())[0])
        self.scheduler = dict_optimizer_scheduler[self.scheduler_name]['class']
        self.scheduler_params = dict_optimizer_scheduler[self.scheduler_name]['params']

        self.device = torch.device(train_params.get('device')) if train_params.get('device') else device
        self.eval_n = max(int(train_params['max_epochs'] * (float(os.environ.get('EVAL_FREQ', 100)) / 100)), 1)
        self.save_path = os.path.join(os.environ["SAVE_PATH"], dataset, dt.now().strftime("%Y-%m-%d"), model)
        self.max_epochs = train_params['max_epochs']
        self.batch_size = train_params['batch_size']

        self.loss = dict_loss[loss](**loss_params, device=self.device)

        self.resume_path = resume_path

        self.writer = None
        self.f_writer = FileWriter

        self.metric_calculator = MetricCalculatorExample
        self.metric_track_key = train_params['metric_track_key']
        self.metric_track_mode = train_params['metric_track_mode']
        assert self.metric_track_mode in ['min', 'max']
        self.best_metric = 0 if self.metric_track_mode == 'max' else 1000

        self.workers = int(os.environ.get('WORKERS', 6))

        seed = 0
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def train(self):
        self.writer = TensorboardWriter(self.model, self.save_path)

        # training and validation data loaders
        dataset_train = self.dataset(**self.data_params, fold=TRAIN)
        train_loader = DataLoader(dataset_train, batch_size=self.batch_size, shuffle=True, num_workers=self.workers)

        dataset_val = self.dataset(**self.data_params, fold=VAL)
        val_loader = DataLoader(dataset_val, batch_size=3, shuffle=False, num_workers=self.workers)

        model, start_epoch = self._init_model()

        self.writer.add_model_params(model)
        for epoch in range(start_epoch, self.max_epochs):
            model.train()
            train_loss, train_loss_comp, train_metrics = self._main_phase(model, train_loader, TRAIN, epoch)
            with torch.no_grad():
                model.eval()
                val_loss, val_loss_comp, val_metrics = self._main_phase(model, val_loader, VAL, epoch)

            if self.scheduler_name in SCHED_STEP_PARAM:
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()

            if epoch % self.eval_n == 0:
                self.writer.add_losses(train_loss, train_loss_comp, val_loss, val_loss_comp, epoch)
                self.writer.add_metrics(val_metrics, VAL, epoch)

            if self._is_best_metric(val_metrics):
                self.writer.add_text("best metrics epoch in validation", str(val_metrics), epoch)
                self._save_model(model, 'best', epoch)

            self._save_model(model, 'last', epoch)
        self.writer.close()

    def test(self, output_path):
        self.writer = FileWriter(self.model_name, self.dataset_name, output_path)

        dataset = self.dataset(**self.data_params, fold=TEST)
        data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=self.workers)

        model, _ = self._init_model()
        total_parameters = sum(p.numel() for p in model.parameters())

        for param in model.parameters():
            param.requires_grad = False

        with torch.no_grad():
            model.eval()
            loss, loss_comp, metrics = self._main_phase(model, data_loader, VAL)

        csv_save_dict = dict(**metrics, name=self.model_name, params=total_parameters)

        self.writer.add_metrics_to_csv(csv_save_dict)

    def classical_methods(self, output_path):
        self.writer = FileWriter(self.model_name, self.dataset_name, output_path)

        dataset_train = self.dataset(**self.data_params, fold=TRAIN)
        data_loader = DataLoader(dataset_train, batch_size=self.batch_size, shuffle=True, num_workers=self.workers)
        metrics = self.metric_calculator(len(data_loader))

        method = self.model

        for idx, batch in enumerate(data_loader):
            input_data = {k: v.to(self.device).squeeze(0).permute(1, 2, 0).numpy() for k, v in batch.items()}
            output_data = method(**input_data)
            input_data = {k: torch.from_numpy(v).permute(2, 0, 1).unsqueeze(0) for k, v in input_data.items()}
            output_data = {k: torch.from_numpy(v).permute(2, 0, 1).unsqueeze(0) for k, v in output_data.items()}
            metrics.add_metrics(**input_data, **output_data)
        csv_save_dict = dict(**metrics.dict, name=self.model_name)
        self.writer.add_metrics_to_csv(csv_save_dict)

    def _save_model(self, model, version, epoch: int = None):
        try:
            os.makedirs(self.save_path + '/ckpt/')
        except FileExistsError:
            pass
        save_path = self.save_path + f'/ckpt/weights_{version}.pth'
        ckpt = {'experiment': self.experiment, 'model': model, 'epoch': epoch,
                'optimizer': self.optimizer.state_dict(), 'scheduler': self.scheduler.state_dict()}
        torch.save(ckpt, save_path)

    def _init_model(self):
        if self.resume_path is not None:
            ckpt = torch.load(self.resume_path, map_location=self.device)
            model = ckpt['model']
            model = model.float()
            model.to(self.device)
            self.optimizer = self.optimizer(model.parameters(), **self.optim_params)
            self.optimizer.load_state_dict(ckpt['optimizer'])
            self.scheduler = self.scheduler(self.optimizer, **self.scheduler_params)
            self.scheduler.load_state_dict(ckpt['scheduler'])
            start_epoch = ckpt.get('epoch', 0) + 1
        else:
            model = self.model(**self.model_params)
            model = model.float()
            model.to(self.device)
            self.optimizer = self.optimizer(model.parameters(), **self.optim_params)
            self.scheduler = self.scheduler(self.optimizer, **self.scheduler_params)
            start_epoch = 0

        return model, start_epoch

    @staticmethod
    def _do_patches(data, ps):
        data_shape = data.size
        assert data_shape(2) % ps == 0 and data_shape(3) % ps == 0
        patches = unfold(data, kernel_size=ps, stride=ps)
        patch_num = patches.size(2)
        patches = patches.permute(0, 2, 1).view(data_shape(0), -1, data_shape(1), ps, ps)
        return torch.reshape(patches, (data_shape(0) * patch_num, data_shape(1), ps, ps))

    @staticmethod
    def _undo_patches(data, n, w, h, ps):
        patches = data.reshape(n, data.size(0), data.size(1), ps, ps)
        patches = patches.view(n, data.size(0), data.size(1) * ps * ps).permute(0, 2, 1)
        return fold(patches, (w, h), kernel_size=ps, stride=ps)

    def _main_phase(self, model, data_loader, phase, epoch=None):
        metrics = self.metric_calculator(len(data_loader))
        epoch_loss = []
        epoch_loss_components = dict()
        self.images = list()
        with tqdm(enumerate(data_loader), total=len(data_loader), leave=True) as pbar:
            for idx, batch in pbar:
                self.optimizer.zero_grad()

                input_data = {k: v.to(self.device) for k, v in batch.items()}

                output_data = model(**input_data)

                loss, loss_components = self.loss(**input_data, **output_data)

                self._images_to_show(input_data, output_data)

                epoch_loss.append(loss.item())
                epoch_loss_components = self._set_loss_components(epoch_loss_components, loss_components)

                if phase == TRAIN:
                    loss.backward()
                    self.optimizer.step()

                metrics.add_metrics(**input_data, **output_data)

                if epoch is not None:
                    pbar.set_description(f'Epoch: {epoch}; {phase} Loss {np.array(epoch_loss).mean():.6f}')

        for k in epoch_loss_components.keys():
            epoch_loss_components[k] = np.array(epoch_loss_components[k]).mean()

        epoch_loss = np.array(epoch_loss).mean()
        if epoch % self.eval_n == 0:
            self.writer.add_images(self.images, phase, epoch)
            self.writer.add_metrics(metrics.dict, phase, epoch)

        return epoch_loss, epoch_loss_components, metrics.dict

    def load_from_dict(self, **ckpt):
        for k, v in ckpt.items():
            setattr(self, k, v)

    def _is_best_metric(self, metric_dict):
        value = metric_dict[self.metric_track_key]
        if self.metric_track_mode == 'min':
            result = self.best_metric > value
            self.best_metric = min(self.best_metric, value)
        else:
            result = self.best_metric < value
            self.best_metric = max(self.best_metric, value)
        return result

    @staticmethod
    def _set_loss_components(epoch_loss_components, loss_components):
        for k in loss_components.keys():
            v = loss_components[k]
            try:
                epoch_loss_components[k].append(v)
            except KeyError:
                epoch_loss_components.update({k: [v]})
        return epoch_loss_components

    def _images_to_show(self, input_data, output_data):
        n_plots = len(input_data) + len(output_data)
        n_images = 20 // n_plots
        if not self.images:
            self.images = list(input_data.items())+list(output_data.items())
        elif self.images[0][1].shape[0] < n_images:
            for i, ten in enumerate(list(input_data.items())+list(output_data.items())):
                self.images[i] = (self.images[i][0], torch.cat([self.images[i][1], ten[1]], dim=0))


class ParseKwargs(argparse.Action):
    CHOICES = dict()

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, self.CHOICES)
        for value in values:
            key, value = value.split('=')
            if self.CHOICES and key not in self.CHOICES.keys():
                print(f"{parser.prog}: warning: argument {option_string}: invalid choice: '{key}' (choose from {list(self.CHOICES.keys())})")
            else:
                getattr(namespace, self.dest)[key] = self._parse(value)

    @staticmethod
    def _parse(data):
        try:
            return int(data)
        except ValueError:
            pass
        try:
            return float(data)
        except ValueError:
            pass
        return data
