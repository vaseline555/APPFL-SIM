# `APPFL-SIM` Configuration Guides

This file lists runtime arguments used by `python -m appfl_sim.runner`, with default values.
Defaults come from `appfl_sim/config/examples/simulation.yaml` and code-side fallbacks.

## Essential Arguments (required)
- `config` (CLI flag, optional): path to YAML configuration file. If omitted, default config is used:
  `appfl_sim/config/examples/simulation.yaml`.
- `exp_name` (default: `appfl-sim`): experiment name; used in output paths.
- `seed` (default: `42`): global random seed.
- `device` (default: `cpu`): device for clients. 
- `server_device` (default: `cpu`): device for server.
- `backend` (default: `serial`): execution backend (`serial`, `nccl`, or `gloo`).
- `dataset` (default: `MNIST`): dataset name.
- `model` (default: `SimpleCNN`): local APPFL model name.
- `num_rounds` (default: `20`): number of federated rounds.
- `num_clients` (default: `20`): number of participating clients.
- `num_sampled_clients` (default: `4`): number of sampled clients per round.
- `update_base` (default: `epoch`): how to treat the local update (`epoch` or `iter`).
- `local_epochs` (default: `1`): (when `update_base=epoch`) local epochs per selected client.
- `local_iters` (default: `1`): (when `update_base=iter`)  local iterations per selected client.
- `batch_size` (default: `32`): local train batch size.
- `lr` (default: `0.01`): learning rate for local updates.
- `optimizer` (default: `SGD`): optimizer from `torch.optim`.
- `criterion` (default: `CrossEntropyLoss`): loss from `torch.nn`.
- `algorithm` (default: `fedavg`): algorithm label used for component inference.
- `trainer` (default: `"vanilla"`): explicit client trainer class name (optional).
- `aggregator` (default: `"fedavg"`): algorithm name used for federated learning simulation (optional).
- `scheduler` (default: `"sync"`): explicit scheduler class name (optional).

## Dataset
- `dataset_dir` (default: `./data`): dataset root path.
- `dataset_loader` (default: `auto`): loader mode (`auto`, `torchvision`, `torchtext`, `torchaudio`, `medmnist`, `leaf`, `flamby`, `tff`, `external`, `custom`).
- `download` (default: `true`): download datasets if missing.
- `train_data_shuffle` (default: `true`): trainer train dataloader shuffle flag.
- `val_data_shuffle` (default: `false`): trainer val dataloader shuffle flag.

### Dataset Split
- `infer_num_clients` (default: `false`): infer client count from fixed-pool datasets (`leaf`, `flamby`, `tff`).
- `split_type` (default: `iid`): split policy (`iid`, `dirichlet`, `pathological`, `unbalanced`).
- `unbalanced_keep_min` (default: `0.5`): minimum keep ratio for unbalanced split.
- `min_classes` (default: `2`): minimum unique classes per client for pathological split.
- `dirichlet_alpha` (default: `0.3`): Dirichlet concentration parameter.

### Custom/External Dataset
- `dataset_external_source` (default: ``): external source (`hf` or `timm`).
- `dataset_external_name` (default: ``): external dataset id/name.
- `dataset_external_config_name` (default: ``): external config/subset name.
- `dataset_external_train_split` (default: `train`): external train split name.
- `dataset_external_test_split` (default: `test`): external test split name.
- `dataset_external_feature_key` (default: ``): feature column/key.
- `dataset_external_label_key` (default: ``): label column/key.
- `dataset_custom_path` (default: ``): path-based custom dataset input.
- `dataset_custom_loader` (default: ``): custom callable path (`module:function`).
- `dataset_custom_kwargs` (default: `{}`): JSON string kwargs for custom loader.

### Benchmark-specific
- `flamby_data_terms_accepted` (default: `false`): acknowledge FLamby data terms.
- `leaf_raw_data_fraction` (default: `1.0`): LEAF raw-data fraction before split.
- `leaf_min_samples_per_client` (default: `2`): LEAF per-client minimum samples.

## Model
- `model_source` (default: `auto`): model backend (`auto`, `appfl`, `timm`, `hf`).
- `model_name` (default: `SimpleCNN`): exact model identifier/name.
- `model_kwargs` (default: `{}`): backend-agnostic model kwargs.
- `model_in_channels` (default: inferred from input shape): explicit input channels override.
- `model_num_classes` (default: inferred from dataset): explicit class-count override.
- `model_timm_pretrained` (default: `false`): timm pretrained flag override.
- `model_timm_kwargs` (default: `{}`): timm kwargs.
- `model_hf_pretrained` (default: `false`): HF pretrained checkpoint loading flag.
- `model_hf_task` (default: `sequence_classification`): HF task type.
- `model_hf_local_files_only` (default: `false`): HF local-only loading.
- `model_hf_trust_remote_code` (default: `false`): HF remote-code trust.
- `model_hf_gradient_checkpointing` (default: `false`): enable HF gradient checkpointing.
- `model_hf_kwargs` (default: `{}`): HF kwargs.
- `model_hf_config_overrides` (default: `{}`): HF config overrides for scratch configs.

### Model Configurations
- `num_layers` (default: `2`): stacked layer count.
- `hidden_size` (default: `64`): hidden dimension for models.
- `seq_len` (default: `128`): sequence length.
- `num_embeddings` (default: `10000`): unique embedding counts.
- `embedding_size` (default: `128`): embedding dimension.
- `use_model_tokenizer` (default: `false`): use model tokenizer in text parser.
- `crop` (default: `28`): image crop shape for augmentation.
- `resize` (default: `28`): image resize shape for augmentation.
- `dropout` (default: `0.0`): dropout probability.

## Optimization
- `weight_decay` (default: `0.0`): optimizer weight decay.
- `max_grad_norm` (default: `0.0`): grad clipping trigger (`>0` enables clipping).
- `eval_batch_size` (default: `128`): batch size for local/global evaluation.
- `num_workers` (default: `0`): `DataLoader` workers per process.
- `pin_memory` (default: auto; `true` on CUDA devices, else `false`): base `DataLoader` pin-memory flag.
- `train_pin_memory` (default: inherits `pin_memory`): train `DataLoader` pin-memory override.
- `eval_pin_memory` (default: inherits `pin_memory`): val/test `DataLoader` pin-memory override.
- `dataloader_persistent_workers` (default: `false`): pass-through to `DataLoader` `persistent_workers` when `num_workers > 0`.
- `dataloader_prefetch_factor` (default: `2`): pass-through to `DataLoader` `prefetch_factor` when `num_workers > 0`.

## Evaluation
- `eval_every` (default: `1`): global/federated evaluation cadence.
- `do_pre_validation` / `do_validation` (default: `true`).
- `show_eval_progress` (default: `true`).
- `enable_global_eval` (default: `true`).
- `enable_federated_eval` (default: `true`).
- `federated_eval_scheme` (default: `holdout_dataset`): federated evaluation mode (`holdout_dataset` or `holdout_client`).
- `holdout_dataset_ratio` (default: `[80,20]`): (when `federated_eval_scheme=holdout_dataset`) local dataset split for each client.  
  (use `[100]` for train-only, `[80,20]` for train/test or `[80,10,10]` for train/val/test, also accepts 1.0 scale).  
  Special case: `[100]` or `[1.0]` means train-only mode (no val/test, no global/federated eval).
- `holdout_client_counts` (default: `0`): (when `federated_eval_scheme=holdout_client`) number of holdout clients.
- `holdout_client_ratio` (default: `0.0`): (when `federated_eval_scheme=holdout_client`) ratio of holdout client size.

### Metrics
- `eval_metrics` (default in code: `[acc1]` when unspecified): metric list.  
  (supported: `acc1`, `acc5`, `auroc`, `auprc`, `youdenj`, `f1`, `precision`, `recall`, `seqacc`, `mse`, `rmse`, `mae`, `mape`, `r2`, `d2`, `dice`, `balacc`)

## Logging
- `log_dir` (default: `./logs`): output log root.
- `log_backend` (default: `file`): tracking backend (`none`, `file`, `console`, `tensorboard`, `wandb`).
- `project_name` (default: `appfl-sim`): project/log directory name (`tensorboard`) or project ID (`wandb`).
- `logging_scheme` (default: `auto`): per-client logging policy (`auto`, `both`, `server_only`).  
  (`auto`: when `num_sampled_clients < num_clients`, per-cleint logging is forced off for performance, i.e., server-only logging)
- `enable_wandb` (default: `false`): enable per-client wandb metric streaming.
- `wandb_mode` (default: `online`): wandb mode (`online` or `offline`).
- `wandb_entity` (default: `""`): wandb entity/team.

## Memory Usage
- `stateful_clients` (default: `false`): keep client states across rounds (`true`; cross-silo FL) or stateless/sporadic clients (`false`; cross-device FL).
- `client_processing_chunk_size` (default: `0`): chunk size for sampled/eval client processing (`<=0` enables auto sizing).
- `on_demand_num_workers` (default: `0`): DataLoader workers for on-demand (stateless) local training client builds.
- `on_demand_workers` (default: alias fallback): backward-compatible alias for `on_demand_num_workers`.
- `on_demand_eval_num_workers` (default: `0`): DataLoader workers for on-demand (stateless) federated evaluation client builds.

## Device and Distributed Backends
- `backend=serial`: single-process baseline (recommended for single-node/single-GPU small runs).
- `backend=nccl`: multi-process multi-GPU runtime via `torch.distributed` + NCCL.
- `backend=gloo`: CPU-oriented multi-process runtime via `torch.distributed` + Gloo.
- `device=cuda` is recommended for `nccl` to map ranks across available GPUs.
- `device=cpu` is recommended for `gloo`.
- GPU subset control for `nccl`: set `CUDA_VISIBLE_DEVICES` before launch.  
  Example: `CUDA_VISIBLE_DEVICES=1,3 appfl-sim --config appfl_sim/config/examples/backend/nccl.yaml`

## Advanced Configurations
- `aggregator_kwargs` (default: `{}`): kwargs forwarded to aggregator class construction.
- `scheduler_kwargs` (default: `{}`): kwargs forwarded to scheduler class construction.
- `per_client_logging_warning_threshold` (default: `50`): warning threshold when per-client logging stays enabled.
- `client_weights_mode` (default: `sample_ratio`): aggregation weighting mode (`uniform`, `sample_ratio`, `adaptive`).
- `optimize_memory` (default: `true`): enable memory-saving cleanup paths in trainer/scheduler/aggregator.
- `use_secure_agg` (default: `false`): toggles secure aggregation related runtime behavior.
- `secure_agg_client_weights_mode` (default: `uniform`): secure-aggregation weighting mode (`uniform`, `sample_ratio`).
- `use_dp` (default: `false`): toggles differential privacy related runtime behavior.
- `dp_mechanism` (default: `laplace`): DP backend selector (`Opacus` path has special on-demand constraints).
- `dp_config` (default: `{}`): DP backend-specific config (e.g., Opacus `noise_multiplier`, `max_grad_norm`).
- `clip_grad` (default: auto from `max_grad_norm`): explicitly enable gradient clipping.
- `clip_norm` (default: `2.0`): gradient norm type used by clip-grad.

## Purged/Deprecated
The following arguments are removed and should not be used:
- `server_lr`
- `server_momentum`
- `local_val_ratio`
- `log_file`
- `leaf_text_token_cache`
