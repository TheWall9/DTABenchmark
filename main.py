from toolbox import Evaluator

if __name__=='__main__':
    Evaluator(model_name='HiMambaDTA', deterministic=True,
              monitor_metric='val/CI', monitor_mode='max',
              use_wandb_logger=True, wandb_project='DTA Benchmark model', wandb_online=False,
              merge_train_val=True, n_res_expand=0, num_workers=10).run()  #