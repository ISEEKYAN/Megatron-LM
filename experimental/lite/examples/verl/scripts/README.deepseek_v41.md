# DeepSeek V4.1 SFT through verl

Requires a checkout with the `deepseek_v41` MLite registry and protocol installed
(the model implementation is delivered separately), with its optimizer declaring
`owns_param_group_policy=True`. Older producers retain legacy scheduling. This preset configures verl's existing
`SFTTrainer` and MLite engine extension with the model-owned Muon optimizer.
It adds no engine capability; `++engine.impl_cfg.optimizer=muon` uses the
existing generic override path. Scheduler/config fixes are shared connector fixes.
The supported example uses PP=TP=CP=EP=1; additional GPUs are data parallel replicas.

```bash
MODEL_PATH=/path/to/hf-checkpoint \
TRAIN_FILES=/path/to/messages.parquet \
VERL_ROOT=/path/to/verl MEGATRON_ROOT=/path/to/Megatron-LM \
NUM_GPUS=2 TRAIN_BATCH_SIZE=4 TOTAL_STEPS=8 LR=1e-4 \
WANDB_ENTITY=megatron-core-moe-dev \
bash experimental/lite/examples/verl/scripts/run_deepseek_v41_sft.sh
```

Authenticate W&B before launching. The default loggers include W&B, console, and
file. Input parquet uses verl's `messages` format and the checkpoint's chat
template. `DRY_RUN=1` prints the resolved launch command without training.

The model retains its per-group learning-rate ratios and weight decay under the
shared scheduler; rejected optimizer updates do not advance that scheduler.
The model-owned policy requires positive base LR and constant weight decay.
Use `LR` to set both recipe LR fields; overriding only one field is not the recipe. Residual dtype,
quantization, and model options can be supplied as Hydra overrides after the
script name, for example `+engine.impl_cfg.quantized=False`.

For synthetic reduced models, the fused verl loss kernel requires hidden size
to be divisible by 128. The engine uses parameter offload to move parameters
and gradients together; independent `GRAD_OFFLOAD=True` is unsupported.

Validation limits: this example has no `application.verl` →
`basic.align_e2e_precision` SFT comparison chain; the default eight-update smoke
is not an e2e precision substitute. The `owns_param_group_policy` validation
surface is real engine initialization (name × ownership), per-group LR/WD through
warmup and scheduler reload, and successful/rejected optimizer transactions.
`router_aux_loss_coef=None` intentionally means `0.0`, matching the RL default.

Opt-in weight fingerprints sample about 1/16 of ordinary tensors (plus selected
important tensors), at most 256 bytes each; stream SHA equality is not full-byte
or bitwise equality. Layerwise reload observability is stderr-only; receiver
fingerprints currently label rank zero. DRY_RUN tests inspect stdout substrings;
Hydra composition checks configuration, not runtime behavior. Missing opaque
registration configuration logs `VERL_MLITE_OPAQUE_CONFIG disabled` and returns
False; existing Transformers registrations still take precedence.
