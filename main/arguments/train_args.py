from sentence_transformers.training_args import BatchSamplers

args_config = {
    "num_train_epochs": 10,
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 8,
    "per_device_eval_batch_size": 4,
    "gradient_checkpointing": True,
    "warmup_ratio": 0.1,
    "learning_rate": 2e-5,
    "lr_scheduler_type": "cosine",
    "optim": "adamw_torch_fused",
    "fp16": True,
    "batch_sampler": BatchSamplers.NO_DUPLICATES,
    "eval_strategy": "steps",
    "save_steps": 500,
    "logging_steps": 100,
    "load_best_model_at_end": True,
    "max_grad_norm": 1.0,
    "metric_for_best_model": "eval_dim_512_cosine_ndcg@10",
    "report_to": ["wandb"],
    "run_name": "experiment_from_config"
}