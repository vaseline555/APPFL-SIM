# `APPFL-SIM` Configuration Guides

This file lists runtime arguments used by `python -m appfl_sim.runner`, with default values.
Defaults come from `appfl_sim/config/examples/simulation.yaml` and code-side fallbacks.

## CLI flag
- `config` (CLI flag, optional): path to YAML configuration file. If omitted, default config is used:
  `appfl_sim/config/examples/simulation.yaml`.

## Experiment (`experiment`)
- `name` (default: `appfl-sim`): experiment name; used in output paths. (prev. `exp_name`)
- `seed` (default: `42`): global random seed.
- `device` (default: `cpu`): device for clients. 
- `server_device` (default: `cpu`): device for server.
- `backend` (default: `serial`): execution backend (`serial`, `nccl`, `gloo`).
- `stateful` (default: `false`): keep client states across rounds (`true`; cross-silo FL) or stateless/sporadic clients (`false`; cross-device FL). (prev `stateful_clients`)

## Dataset (`dataset`)
- `path` (default: `./data`): dataset root path. (prev. `dataset_dir`)
- `name` (default: `MNIST`): dataset name. (prev. `dataset`)
- `backend` (default: `torchvision`): loader mode (`torchvision`, `torchtext`, `torchaudio`, `medmnist`, `leaf`, `flamby`, `tff`, `hf`, `custom`). (prev. `dataset_loader`)
- `download` (default: `true`): download datasets if missing.
- `load_dataset` return contract: `(client_datasets, server_dataset, dataset_meta)`.
- `configs` (default: `{}`): dataset-agnostic keyword arguments.
  - `raw_data_fraction` (default: `1.0`): raw-data fraction before split for LEAF benchmark.
  - `min_samples_per_client` (default: `2`): minimum samples per client for LEAF benchmark.
  - `terms_accepted` (default: `true`): acknowledge FLamby data terms.

## Split Simulation (`split`)
- `type` (default: `iid`): split policy (`iid`, `unbalanced`, `dirichlet`, `pathological`, `pre`). (prev. `split_type`)
  - `pre`: use benchmark-defined client partition (required for `dataset.backend` in `leaf`, `flamby`, `tff`).
- `infer_num_clients` (default: `false`): infer client count from fixed-pool datasets (`leaf`, `flamby`, `tff`).
- `configs`: split type-agnostic keyword arguments.
  - `unbalanced_keep_min` (default: `0.5`): minimum keep ratio (when `split.type=unbalanced`).
  - `dirichlet_alpha` (default: `0.3`): concentration parameter (when `split.type=dirichlet`).
  - `min_classes` (default: `2`): minimum unique classes per client (when `split.type=pathological`).

## Model (`model`)
- `path` (default: `./models`): model root path.
- `name` (default: `SimpleCNN`): model name. (prev. `model`)
- `backend` (default: `auto`): model source (`auto`, `local`, `hf`, `torchvision`, `torchtext`, `torchaudio`).
- `configs` (default: `{}`): backend-agnostic model keyword arguments.
  - `in_channels` (default: inferred from input shape): explicit input channels override.
  - `hidden_size` (default: `64`): hidden dimension for models.
  - `num_classes` (default: inferred from dataset): explicit class-count override.
  - `num_layers` (default: `2`): stacked layer count.
  - `dropout` (default: `0.0`): dropout probability.
  - `seq_len` (default: `128`): sequence length.
  - `num_embeddings` (default: `10000`): unique embedding counts.
  - `embedding_size` (default: `128`): embedding dimension.
  - `use_model_tokenizer` (default: `false`): use model tokenizer in text parser.

## Training (`train`)
- `num_rounds` (default: `20`): number of federated rounds.
- `num_clients` (default: `20`): number of participating clients.
- `num_sampled_clients` (default: `4`): number of sampled clients per round.
- `update_base` (default: `epoch`): how to treat the local update (`epoch` or `iter`).
- `local_epochs` (default: `1`): (when `update_base=epoch`) local epochs per selected client.
- `local_iters` (default: `1`): (when `update_base=iter`)  local iterations per selected client.
- `batch_size` (default: `32`): local train batch size.
- `shuffle` (default: `true`): trainer train dataloader shuffle flag. (prev. ``train_data_shuffle`)
- `max_grad_norm` (default: `0.0`): grad clipping trigger (`>0` enables clipping).
- `eval_batch_size` (default: `128`): batch size for local/global evaluation.
- `num_workers` (default: `0`): `DataLoader` workers per process.
- `pin_memory` (default: auto; `true` on CUDA devices, else `false`): base `DataLoader` pin-memory flag.
- `train_pin_memory` (default: inherits `pin_memory`): train `DataLoader` pin-memory override.
- `eval_pin_memory` (default: inherits `pin_memory`): val/test `DataLoader` pin-memory override.
- `dataloader_persistent_workers` (default: `false`): pass-through to `DataLoader` `persistent_workers` when `num_workers > 0`.
- `dataloader_prefetch_factor` (default: `2`): pass-through to `DataLoader` `prefetch_factor` when `num_workers > 0`.

## Algorithm (`algorithm`)
- `name` (default: `fedavg`): algorithm label used for component inference.
- `mix_coefs` (default: `sample_ratio`): aggregation coefficients (`uniform`, `sample_ratio`, `adaptive`).
- `optimize_memory` (default: `true`): enable memory-saving cleanup paths in trainer/scheduler/aggregator.
- `aggregator` (default: `""`): explicit aggregator class name (optional).
- `scheduler` (default: `""`): explicit scheduler class name (optional).
- `trainer` (default: `""`): explicit trainer class name (optional).
- `aggregator_kwargs` (default: `{}`): configs forwarded to aggregator class construction.
- `scheduler_kwargs` (default: `{}`): configs forwarded to scheduler class construction.
- `trainer_kwargs` (default: `{}`): configs forwarded to trainer class construction.

## Optimization (`optimization`)
- `optimizer` (default: `SGD`): optimizer from `torch.optim`.
- `criterion` (default: `CrossEntropyLoss`): loss from `torch.nn`.
- `lr` (default: `0.01`): learning rate for local updates.
- `lr_decay` (default: disabled): local LR scheduler applied during each client update.
  - `enable` (default: `false`): enable local LR decay.
  - `type` (default: `none`): scheduler type (`none`, `exponential`, `cosine`).
  - `gamma` (default: `0.99`): exponential decay factor when `type=exponential`.
  - `t_max` (default: `0`): cosine cycle length when `type=cosine`; `<=0` auto-uses local update length.
  - `eta_min` (default: `0.0`): cosine minimum LR when `type=cosine`.
  - `min_lr` (default: `0.0`): floor clamp applied after scheduler step.
- `clip_grad_norm` (default: `0.0`): enable gradient norm clipping when `>0`.
- `accum_grad` (default: `0`): accumulate gradient when `>0`.
- `configs` (default: `{}`): optimizer-agnostic keyword arguments.
  - `weight_decay` (default: `0.0`): weight decay.
  - `momentum` (default: `0.0`): moemntum.

## Evaluation (`eval`)
- `every` (default: `1`): global/federated evaluation cadence.
- `metrics` (default in code: `[acc1]` when unspecified): metric list.  
  (supported: `acc1`, `acc5`, `auroc`, `auprc`, `youdenj`, `f1`, `precision`, `recall`, `seqacc`, `mse`, `rmse`, `mae`, `mape`, `r2`, `d2`, `dice`, `balacc`)
- `do_pre_evaluation` / `do_post_evaluation` (default: `true`): flag for evaluation before/after local updates.
- `show_eval_progress` (default: `true`): flag for showing tqdm progress bar during global/federated evaluation.
- `enable_global_eval` (default: `true`): flag for server-side global evaluation if server-side holdout set exists.
- `enable_federated_eval` (default: `true`): flag for client-side federated evaluation if each client has holdout set.
- `configs` (default: `{}`): configurations for federated evaluation.
  - `scheme` (default: `dataset`): federated evaluation mode (`dataset` or `client`).
  - `dataset_ratio` (default: `[80,20]`): local dataset split ratio for each client when `eval.configs.scheme=dataset`.  
    (e.g., `[100]` for train-only, `[80,20]` for train/test or `[80,10,10]` for train/val/test, also accepts 1.0 scale).  
    Special case: `[100]` or `[1.0]` means train-only mode (no val/test, no global/federated eval).
  - `client_ratio` (default: `0.0`): ratio of holdout clients size when `eval.configs.scheme=client`.
  - `client_counts` (default: `0`): number of holdout clients when `eval.configs.scheme=client`.

## Logging (`logging`)
- `path` (default: `./logs`): output log path.
- `name` (default: `appfl-sim`): ignored at runtime; logging project/name is forced to `experiment.name`.
- `backend` (default: `file`): logging scheme (`none`, `file`, `console`, `tensorboard`, `wandb`).
- `type` (default: `auto`): logging policy (`auto`, `both`, `server_only`).  
  (`auto`: when `num_sampled_clients < num_clients`, per-cleint logging is forced off for performance, i.e., server-only logging)
- `configs` (default: `{}`): backend-agnostic keyword arguments.
  - `wandb_mode` (default: `online`): wandb mode (`online` or `offline`).
  - `wandb_entity` (default: `""`): wandb entity/team.
  - `track_gen_rewards` (default: `false`): log per-round and cumulative generalization reward (`-(g_t - g_{t-1})`) computed at server from round global generalization error.

## Privacy (`privacy`)
- `use_dp` (default: `false`): toggles differential privacy related runtime behavior.
- `mechanism` (default: `laplace`): DP backend selector (`Opacus` path has special on-demand constraints).
- `clip_grad_norm` (default: `0.0`): enable gradient norm clipping when `>0`.
- `clip_norm_type` (default: `2.0`): gradient norm type used by gradient norm clipping.
- `kwargs` (default: `{}`): DP backend-specific keyword arguments (e.g., Opacus `noise_multiplier`, `max_grad_norm`).

## Secure Aggregation (`secure_aggregation`)
- `use_sec_agg` (default: `false`): toggles secure aggregation related runtime behavior.
- `mix_coefs` (default: `uniform`): secure-aggregation weighting mode (`uniform`, `sample_ratio`).
