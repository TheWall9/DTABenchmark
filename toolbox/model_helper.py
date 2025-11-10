import copy
import os
import inspect
from typing import Any

from argparse import Namespace
import torch
from torch import nn, optim
import wandb
import lightning
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from lightning.pytorch.callbacks import RichProgressBar, TQDMProgressBar, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.utilities.types import STEP_OUTPUT

from toolbox.datamodule import DataModule, DTADatasetBase
from toolbox.metrics import DTAMetrics
from toolbox.config import LIGHTNING_LOGS_DIR

class ModelBase(lightning.LightningModule):
    datamodule_cls = DataModule
    dataset_cls = DTADatasetBase
    def __init__(self, config=None):
        super().__init__()
        self.config = config
        self.epoch_train_outputs = []
        self.epoch_val_outputs = []
        self.epoch_test_outputs = []
        self.train_metric = DTAMetrics()
        self.val_metric = DTAMetrics()
        self.test_metric = DTAMetrics()
        self.metric_history = {}
        self.label_keys = ['affinity']
        self.forward_args_name = inspect.getfullargspec(self.forward).args[1:]
        self.step_args_name = inspect.getfullargspec(self.step).args[1:]
        self.save_hyperparameters(config, logger=False)

    @classmethod
    def get_valid_inputs_name(self):
        step_args = inspect.getfullargspec(self.step).args[1:]
        forward_args = inspect.getfullargspec(self.forward).args[1:]
        ans = sorted(set(step_args).union(forward_args))
        return ans

    def prepare_foward_inputs(self, batch):
        return {key:batch[key] for key in self.forward_args_name if key in batch}

    @property
    def metric(self):
        return getattr(self, f'{self.current_stage}_metric')

    @property
    def current_stage(self):
        stage = self.trainer.state.stage.value
        stage_map = {"validate":"val", "sanity_check": 'val'}
        return stage_map.get(stage, stage)

    @property
    def is_log_epoch(self):
        stage = self.current_stage
        if stage =="train":
            interval = self.config[f'check_{stage}_every_n_epoch']
            return (self.current_epoch+1)%interval==0 if interval>0 else False
        elif stage in ["predict", 'sanity_check']:
            return False
        return True

    def on_batch_end_fn(self, outputs, batch):
        if self.is_log_epoch:
            self.metric.update(outputs['predict'], batch['affinity'])
        loss_keys = [key for key in outputs if key.startswith('loss')]
        for key in loss_keys:
            self.log(f"{self.current_stage}/{key}", outputs[key], on_step=True, on_epoch=True, prog_bar=True)

    def on_train_batch_end(self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int) -> None:
        self.on_batch_end_fn(outputs, batch)

    def on_validation_batch_end(self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        self.on_batch_end_fn(outputs, batch)

    def on_test_batch_end(self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        self.on_batch_end_fn(outputs, batch)

    def on_epoch_end_fn(self, hp_metric=True):
        tag = self.current_stage
        self.log("step", self.current_epoch, on_step=False, on_epoch=True)
        if self.is_log_epoch:
            metric = self.metric.compute()
            metric = {f"{tag}/{key}":value for key,value in metric.items()}
            self.log_dict(metric, on_epoch=True)

    def on_train_epoch_end(self):
        self.on_epoch_end_fn()
        self.metric_history.update({key:value.item() for key,value in self.trainer.logged_metrics.items()})
        self.metric_history['last_epoch'] = self.current_epoch
        if self.current_epoch == 0:
            hparams = self._convert_hyperparams(self.hparams)
            # metrics = self.trainer.logged_metrics
            metrics = self.metric_history
            # metrics = {f"hparam/{key}":value for key, value in metrics.items()}
            self.logger.log_hyperparams(hparams, metrics)
            old_haparams_file = os.path.join(self.logger.log_dir, 'hparams.yaml')
            if os.path.exists(old_haparams_file):
                os.remove(old_haparams_file)
            self.logger.save()


    def on_validation_epoch_end(self):
        """
        为了self.logger.log_hyperparams(hparams, metrics)生效，self.save_hyperparameters(logger=False)
        由于logger为False导致hparams.yaml为空，重新生成hparams.ymal
        """
        # self.logger.log_metrics()
        self.on_epoch_end_fn()
        self.metric_history.update({key:value.item() for key,value in self.trainer.logged_metrics.items()})


    def on_test_epoch_end(self):
        self.on_epoch_end_fn()
        self.metric_history.update({key:value.item() for key,value in self.trainer.logged_metrics.items()})
        torch.save({"preds": torch.cat(self.test_metric.preds).cpu(),
                    "targets": torch.cat(self.test_metric.targets).cpu(),},
                   os.path.join(self.logger.log_dir, 'preds.pt'))

    def attach_logger(self):
        existed_logger_cls = [logger.__class__ for logger in self.trainer.loggers]
        config = self.config
        version = self.logger.version
        save_dir = self.logger.save_dir
        use_wandb = config.get('use_wandb_logger', False)
        use_csv = config.get('use_csv_logger', False)
        if use_csv and CSVLogger not in existed_logger_cls:
            self.trainer.loggers.append(CSVLogger(save_dir=save_dir, version=version))
        if use_wandb and WandbLogger not in existed_logger_cls:
            online = config.get('wandb_online', False)
            project = config.get('wandb_project', 'wandb_project')
            wandb_save_dir = config.get('wandb_save_dir', self.logger.log_dir)
            if not os.path.exists(wandb_save_dir):
                os.makedirs(wandb_save_dir)
            if wandb.run is not None:
                wandb.finish()
            self.trainer.loggers.append(WandbLogger(save_dir=wandb_save_dir,
                                               offline=not online,
                                               project=project,
                                               config=config,
                                               save_code=True))
            # wandb.run.config.update(config, allow_val_change=True)

    def on_fit_start(self) -> None:
        super().on_fit_start()
        self.attach_logger()
        self.train_metric.reset()

    def auto_step(self, batch):
        return self.step(**{key:batch[key] for key in batch if key in self.step_args_name})

    def step(self, **kwargs):
        raise NotImplementedError

    def on_train_epoch_start(self) -> None:
        self.epoch_train_outputs = []
        self.train_metric.reset()

    def on_validation_epoch_start(self) -> None:
        self.epoch_validation_outputs = []
        self.val_metric.reset()

    def on_test_epoch_start(self) -> None:
        self.epoch_test_outputs = []
        self.test_metric.reset()

    def _convert_hyperparams(self, hparams):
        ans = {}
        for key, value in hparams.items():
            flag = True
            for dtype in (int, str, float, bool):
                if isinstance(value, dtype):
                    flag = False
                    break
            if flag:
                value = str(value)
            ans[key] = value
        return ans

    def _detach_outputs(self, outputs):
        if isinstance(outputs, dict):
            return {
                k: v.detach().cpu() if isinstance(v, torch.Tensor) else v
                for k, v in outputs.items()
            }
        elif isinstance(outputs, torch.Tensor):
            return outputs.detach().cpu()
        return outputs

    def training_step(self, batch, batch_idx=None):
        return self.auto_step(batch)

    def validation_step(self, batch, batch_idx=None):
        return self.training_step(batch, batch_idx=batch_idx)

    def test_step(self, batch, batch_idx=None):
        return self.validation_step(batch, batch_idx=batch_idx)

    @classmethod
    def add_parser_arguments(cls, parser):
        cls.datamodule_cls.add_parser_arguments(parser)
        cls.dataset_cls.add_parser_arguments(parser)
        parser.add_argument('--lr', default=1e-3, type=float)
        parser.add_argument("--lr_scale", type=float, default=0.5)
        parser.add_argument("--lr_gamma", type=float, default=0.995)
        parser.add_argument('--max_epochs', default=100, type=int)
        parser.add_argument('--model_seed', default=42, type=int)
        parser.add_argument('--check_val_every_n_epoch', default=2, type=int)
        parser.add_argument('--check_train_every_n_epoch', default=10, type=int)
        parser.add_argument('--accumulate_grad_batches', default=1, type=int)
        parser.add_argument('--log_root_dir', default=LIGHTNING_LOGS_DIR, type=str)
        parser.add_argument("--comment", default="debug", type=str)
        parser.add_argument("--deterministic", action="store_true")
        parser.add_argument('--benchmark', action="store_true")
        parser.add_argument('--use_csv_logger', action="store_true")
        parser.add_argument('--use_wandb_logger', action="store_true")
        parser.add_argument('--wandb_project', default="wandb_project", type=str)
        parser.add_argument('--wandb_online', action="store_true")
        parser.add_argument('--monitor_metric', type=str, default='val/loss_epoch')
        parser.add_argument('--monitor_mode', type=str, default='min')

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.config['lr'])
        return optimizer

    @classmethod
    def configure_trainer(cls, config, callbacks=None):
        callbacks = [] if callbacks is None else callbacks
        callbacks.append(LearningRateMonitor())
        checkpoint_callback = ModelCheckpoint(
            monitor=config['monitor_metric'],
            mode=config['monitor_mode'],
            save_top_k=1,
            save_last=True,
            filename="best-model-{epoch}-{step}"
        )
        callbacks.append(checkpoint_callback)
        default_root_dir = os.path.join(config['log_root_dir'], config['comment'], config['dataset_full_name'], cls.__name__)
        trainer = lightning.Trainer(max_epochs=config['max_epochs'],
                                    check_val_every_n_epoch=config['check_val_every_n_epoch'],
                                    deterministic=config['deterministic'],
                                    benchmark=config['benchmark'],
                                    accumulate_grad_batches=config['accumulate_grad_batches'],
                                    callbacks=callbacks,
                                    default_root_dir=default_root_dir,
                                    )
        return trainer
