# `APPFL-SIM` Config Arguments

This file lists runtime arguments used by `python -m appfl_sim.runner`, with default values.
Defaults come from `appfl_sim/config/examples/simulation.yaml` and code-side fallbacks.

## Core
- `backend` (default: `mpi`): execution backend (`serial` or `mpi`).
- `exp_name` (default: `appfl-sim`): experiment id used in output paths.
- `seed` (default: `42`): global random seed.
- `algorithm` (default: `fedavg`): algorithm label for run metadata/logging.

## Dataset Selection
- `dataset` (default: `MNIST`): dataset name.
- `dataset_loader` (default: `auto`): loader mode (`auto`, `torchvision`, `torchtext`, `torchaudio`, `medmnist`, `leaf`, `flamby`, `tff`, `custom`, `external`).
- `download` (default: `true`): download datasets if missing (when supported).
- `data_dir` (default: `./data`): dataset root path.

## Custom/External Dataset
- `custom_dataset_loader` (default: `""`): python callable path (`module:function`) for custom loader.
- `custom_dataset_kwargs` (default: `"{}"`): JSON string kwargs for custom loader.
- `custom_dataset_path` (default: `""`): path-based custom dataset input.
- `external_source` (default: `""`): external source (`hf` or `timm`) when using `external` mode.
- `external_dataset_name` (default: `""`): external dataset id/name.
- `external_dataset_config_name` (default: `""`): config/subset name for external dataset.
- `external_train_split` (default: `train`): external train split name.
- `external_test_split` (default: `test`): external test split name.
- `external_feature_key` (default: `""`): feature column/key.
- `external_label_key` (default: `""`): label column/key.
- `flamby_data_terms_accepted` (default: `false`): FLamby terms acknowledgement gate.

## Model Selection
- `model_name` (default: `SimpleCNN`): local APPFL model name (used when `model.name` is empty).
- `model.source` (default: `auto`): model backend (`auto`, `appfl`, `timm`, `hf`).
- `model.name` (default: `""`): exact backend model name/card.
- `model.pretrained` (default: `false`): use pretrained weights.
- `model.kwargs` (default: `{}`): backend-agnostic model kwargs.
- `model.appfl.kwargs` (default: `{}`): local APPFL model kwargs.
- `model.timm.pretrained` (default: `false`): timm pretrained flag override.
- `model.timm.kwargs` (default: `{}`): timm kwargs.
- `model.hf.task` (default: `sequence_classification`): HF task type.
- `model.hf.local_files_only` (default: `false`): HF local-only loading.
- `model.hf.trust_remote_code` (default: `false`): HF remote-code trust.
- `model.hf.gradient_checkpointing` (default: `false`): enable HF gradient checkpointing.
- `model.hf.kwargs` (default: `{}`): HF kwargs.

## FL Scale
- `num_clients` (default: `20`): logical total number of clients.
- `num_rounds` (default: `20`): number of global rounds.
- `num_sampled_clients` (default: `4`): number of sampled clients per round.
- `test_size` (default: `0.2`): loader-level train/holdout split ratio when parser supports it.
- `dataset_split_ratio` (default: unset): optional local split override for each client dataset.
  Use `[80,20]` for train/test or `[80,10,10]` for train/val/test (also accepts 1.0 scale).

## Optimization
- `local_epochs` (default: `1`): local epochs per selected client.
- `batch_size` (default: `32`): local train batch size.
- `eval_batch_size` (default: `128`): evaluation batch size.
- `optimizer` (default: `SGD`): torch optimizer class name.
- `criterion` (default: `CrossEntropyLoss`): torch loss function class name.
- `lr` (default: `0.01`): optimizer learning rate.
- `weight_decay` (default: `0.0`): optimizer weight decay.
- `max_grad_norm` (default: `0.0`): grad clipping trigger (>0 enables clipping path).

## Data Split
- `split_type` (default: `iid`): split policy (`iid`, `dirichlet`, `pathological`, `unbalanced`).
- `dirichlet_alpha` (default: `0.3`): alpha for dirichlet split.
- `min_classes` (default: `2`): min classes for pathological split.
- `unbalanced_keep_min` (default: `0.5`): min keep ratio for unbalanced split.

## Evaluation
- `eval_every` (default: `1`): server/global eval frequency.
- `enable_global_eval` (default: `true`): enable server-side global eval.
- `enable_federated_eval` (default: `true`): enable checkpointed federated eval on local test splits.
- `federated_eval_scheme` (default: `holdout_dataset`): federated eval mode (`holdout_dataset` or `holdout_client`).
- `holdout_eval_num_clients` (default: `0`): number of holdout clients (holdout-client mode).
- `holdout_eval_client_ratio` (default: `0.0`): ratio-based holdout client size.
- `show_eval_progress` (default: `true`): show tqdm progress bars for global/federated evaluation.

## Client Lifecycle and Memory
- `stateful_clients` (default: `false`): keep client objects across rounds (`true`) or instantiate/release on-demand (`false`).
- `client_processing_chunk_size` (default: `0`): chunk size for sampled/eval client-id processing (`<=0` enables auto sizing).
- `offload_to_cpu_after_local_job` (default: `true`): offload model/loss to CPU after local train/eval to reduce VRAM.
- `clear_cuda_cache_after_chunk` (default: `false`): force `torch.cuda.empty_cache()` after each processed chunk (usually slower; keep false unless needed).

## Device and MPI
- `device` (default: `cpu`): client device.
- `server_device` (default: `cpu`): server device.
- `num_workers` (default: `0`): DataLoader workers per process.
- `mpi_dataset_download_mode` (default: `rank0`): dataset download policy (`rank0`, `local_rank0`, `all`, `none`).
- `mpi_use_local_rank_device` (default: `true`): map device by local rank in MPI.
- `mpi_log_rank_mapping` (default: `false`): print rank/device mapping in MPI workers.
- `mpi_num_workers` (default: `0`): MPI worker ranks (0 = auto).
- `mpi_oversubscribe` (default: `false`): pass `--oversubscribe` to MPI launcher.

## Fixed-Pool Client Selection (LEAF/FLamby/TFF)
- `infer_num_clients` (default: `false`): infer from dataset pool when `true`.
- `client_subsample_num` (default: `0`): hard cap of selected clients from fixed pool.
- `client_subsample_ratio` (default: `1.0`): ratio-based subsampling from fixed pool.
- `client_subsample_mode` (default: `random`): subsample mode (`random`, `first`, `last`).
- `client_subsample_seed` (default: `42`): subsample RNG seed.

## Logging and Tracking
- `log_dir` (default: `./logs`): output log root.
- `logging_backend` (default: `file`): tracking backend (`none`, `file`, `console`, `tensorboard`, `wandb`).
- `client_logging_scheme` (default: `auto`): per-client logging policy (`auto`, `per_client`, `aggregated`).
- Runtime rule: when `num_sampled_clients < num_clients`, client file logging is forced off for performance (server-only logging).
- `per_client_logging_threshold` (default: `10`): `auto` cutoff.
- `per_client_logging_warning_threshold` (default: `50`): warning threshold for forced per-client logs.
- `aggregated_logging_scheme` (default: `server_only`): aggregated mode (currently `server_only`).
- `project_name` (default: `appfl-sim`): project/log directory name (tensorboard) or project id (wandb).
- `experiment_name` (default: `appfl-sim`): run name.
- `wandb_entity` (default: `""`): wandb entity/team.
- `wandb_mode` (default: `online`): wandb mode (`online` or `offline`).

## Metrics
- `eval_metrics` (default in code: `[acc1]` when unspecified): metric list.
- `default_eval_metric` (default in code: `acc1` when unspecified): primary metric key.
- `do_pre_validation` (default in code: `true`): local pre-eval on available local val/test splits for sampled clients.
- `do_validation` (default in code: `true`): local post-eval on available local val/test splits for sampled clients.
- `local_gen_error` (derived): reported when local val exists, computed as `post_val_loss - train_loss`.
- `pre/post local eval` vs `federated eval`:
  pre/post is per-round sampled-client diagnostics; federated eval is all-client checkpoint reporting at `eval_every`.

## Model Shape/Text Parameters
- `resize` (default: `28`): image resize hint for local models.
- `crop` (default: `28`): image crop hint for local models.
- `hidden_size` (default: `64`): hidden dimension for sequence models.
- `dropout` (default: `0.0`): dropout probability.
- `num_layers` (default: `2`): stacked layer count.
- `num_embeddings` (default: `10000`): vocabulary size.
- `embedding_size` (default: `128`): embedding dimension.
- `seq_len` (default: `128`): sequence length.
- `use_model_tokenizer` (default: `false`): use model tokenizer in text parser.
- `use_pt_model` (default: `false`): legacy local-model pretrained flag.
- `is_seq2seq` (default: `false`): sequence-to-sequence output mode.
- `need_embedding` (default: `true`): whether model expects token embeddings.

## Dataset-Specific Optional Keys (code defaults)
- `leaf_raw_data_fraction` (default: `1.0`): fraction of LEAF raw pool to keep.
- `leaf_min_samples_per_client` (default: `2`): LEAF per-client min samples.
- `leaf_image_root` (default: `""`): optional LEAF image root override.

## Deprecated Alias
- `effective` / `client_init_mode` (alias): mapped to `stateful_clients` for backward compatibility.
- `client_fraction` (alias): converted to `num_sampled_clients` using `int(client_fraction * num_clients)`.

## Purged (removed from default config)
- `server_lr`: removed (unused by current runner pipeline).
- `server_momentum`: removed (unused by current runner pipeline).
- `local_val_ratio`: removed (unused by current runner path).
- `log_file`: removed (unused by current logger/tracker path).
