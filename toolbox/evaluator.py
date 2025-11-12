import os
import json
import time
import copy
import importlib
import argparse
import optuna
import wandb
import torch
import lightning
import pandas as pd
from datasets.fingerprint import Hasher
from optuna.integration import PyTorchLightningPruningCallback

from toolbox.utils import load_class
from toolbox.config import EVALUATION_TEMP_DIR

class Evaluator():
    def __init__(self, model_name=None, **kwargs):
        self.model_name = model_name
        self.kwargs = kwargs

    def get_model_cls(self, model_name):
        try:
            cls = load_class("benchmark", model_name)
        except:
            cls = load_class("models", model_name)
        return cls

    def parser_args(self, model_name, **kwargs):
        parser = argparse.ArgumentParser([])
        parser.add_argument("--job_dir", default=EVALUATION_TEMP_DIR, type=str)
        if model_name is None:
            parser.add_argument("--model_name", default='DeepDTA', type=str, choices=['DeepDTA'])
            args, _ = parser.parse_known_args()
            model_name = args.model_name
        model_cls = self.get_model_cls(model_name)
        model_cls.add_parser_arguments(parser)
        args = parser.parse_args()
        args = vars(args)
        args.update(kwargs)
        args['model_name'] = model_name
        return args

    def task_generator(self, args):
        model_cls = self.get_model_cls(args['model_name'])
        dataset = model_cls.datamodule_cls(args)
        dataset.prepare_data()
        dataset.setup()
        for fold_id, dataset in enumerate(dataset.split_datasets(merge_train_val=args['merge_train_val'])):
            config = copy.deepcopy(args)
            config.update(dataset.data_info)
            yield config, dataset, model_cls

    def exec(self, config, dataset, model_cls, callbacks=None):
        start_time_stamp = time.time()
        lightning.seed_everything(config['model_seed'])
        model = model_cls(config)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("trainable params:", trainable_params)
        trainer = model_cls.configure_trainer(config, callbacks=callbacks)
        trainer.fit(model, dataset)
        end_time_stamp = time.time()
        if trainer.state.status!='finished':
            trainer._teardown()
            exit(0)
        hparams = model.hparams.copy()
        hparams['fit_time_cost'] = end_time_stamp - start_time_stamp
        version = os.path.basename(trainer.log_dir)
        hparams['version'] = version
        hparams.update(model.metric_history)
        print("load model from ", trainer.checkpoint_callback.best_model_path)
        model = model_cls.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
        trainer.test(model, dataset)
        hparams.update(model.metric_history)
        if wandb.run is not None:
            hparams['wandb_id'] = wandb.run.id
            wandb.finish()
        torch.cuda.empty_cache()
        with open(os.path.join(trainer.log_dir, 'metrics.json'), 'w') as f:
            json.dump(hparams, f, indent=4)
        return os.path.abspath(trainer.log_dir)

    def run(self, debug=False, overwrite=False, callbacks=None):
        args = self.parser_args(self.model_name, **self.kwargs)
        job_dir = os.path.join(args.pop('job_dir'), args['dataset_name'])
        cv_fold_id = args.pop('cv_fold_id', None)
        if not os.path.exists(job_dir):
            os.makedirs(job_dir)
        jobs = {item.split("_")[-1].split(".json")[0]: item for item in os.listdir(job_dir) if item.endswith('.json')}
        job_hash = Hasher.hash(args)
        job_file = os.path.join(job_dir, jobs.get(job_hash, f"running_{job_hash}.json"))
        job_record = {}
        if os.path.exists(job_file):
            with open(job_file) as f:
                job_record = json.load(f)
        print("begin job", job_hash, args)
        for config, dataset, model_cls in self.task_generator(args):
            task_hash = Hasher.hash(config)
            is_valid = task_hash in job_record and job_record[task_hash] is not None
            if is_valid and os.path.exists(job_record[task_hash]) and not overwrite:
                print(f"skip task {task_hash} \n{job_record[task_hash]}")
                continue
            if cv_fold_id is not None and str(config['cv_fold_id'])!=str(cv_fold_id):
                print(f"skip cv_fold_id:{config['cv_fold_id']}")
                continue
            print("exec task", task_hash, config)
            res_path = self.exec(config, dataset, model_cls, callbacks=callbacks)
            if debug:
                return res_path
            print("task done", task_hash)
            job_record[task_hash] = res_path
            with open(job_file, 'w') as f:
                json.dump(job_record, f, indent=4)
        done_job_file = os.path.join(job_dir, f"done_{len(job_record)}_{job_hash}.json")
        done_job_metric_file = os.path.join(job_dir, f"done_{len(job_record)}_{job_hash}.csv")
        if not os.path.exists(done_job_file):
            os.rename(job_file, done_job_file)
        if not os.path.exists(done_job_metric_file):
            ans = []
            for task_dir in job_record.values():
                with open(os.path.join(task_dir, 'metrics.json')) as f:
                    metric_data = json.load(f)
                    ans.append(metric_data)
            ans = pd.DataFrame(ans)
            ans.to_csv(done_job_metric_file, index=False)
        ans = pd.read_csv(done_job_metric_file)
        metric_columns = [col for col in ans.columns if col.startswith("train") or col.startswith("val") or col.startswith("test")]
        metric_data = ans[metric_columns]
        print("result file", os.path.abspath(done_job_metric_file))
        print(metric_data.mean())

    @classmethod
    def optuna_search(cls, study_name, metric_key, direction, hparam_fn, **kwargs):
        STORAGE_PATH = f"sqlite:///{study_name}_study.db"

        def objective(trial):
            hparams = hparam_fn(trial)
            evaluator = cls(**kwargs, **hparams)
            callbacks = [PyTorchLightningPruningCallback(trial, monitor=metric_key)]
            log_path = evaluator.run(debug=True, callbacks=callbacks)
            metric_file = os.path.join(log_path, 'metrics.json')
            with open(metric_file) as f:
                data = json.load(f)
            return data[metric_key]

        try:
            # 尝试加载已有的研究
            study = optuna.load_study(
                study_name=study_name,
                storage=STORAGE_PATH
            )
            print(f"已加载现有研究，当前已完成 {len(study.trials)} 次试验")
        except KeyError:
            # 若研究不存在，则创建新研究
            study = optuna.create_study(
                study_name=study_name,
                storage=STORAGE_PATH,
                direction=direction,
                pruner=optuna.pruners.MedianPruner(),
                sampler=optuna.samplers.TPESampler(),
                load_if_exists=False  # 首次创建时设为False
            )
            print("创建新研究")

        study.optimize(objective, show_progress_bar=True)

        print(f"最佳损失: {study.best_value:.4f}")
        print("最佳参数组合:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")



if __name__=="__main__":
    Evaluator(deterministic=True, max_epochs=1, root_data_dir='../data').run(debug=True)

