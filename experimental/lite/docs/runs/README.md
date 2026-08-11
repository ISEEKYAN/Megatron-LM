# DeepSeek-V4 QAT proxy run

`run_ds4_qat_proxy.sh` is a generic, one-node eight-GPU recipe for the
DeepSeek-V4 DAPO path. It uses the qualified base image, vLLM 0.25.1,
MCore commit `43124b60c`, VERL commit `b9c513c4`, and a ModelOpt side site
that includes cudnn-frontend 1.27 or later. Set the paths required at the top
of the script and submit it with `bash experimental/lite/docs/runs/run_ds4_qat_proxy.sh`.

The recipe binds `ENABLE_QAT=True` to `ROLLOUT_WEIGHT_BITS=4`; the inner DAPO
launcher rejects any QAT/MXFP4 mismatch. Set `DS4_SHARED_DATA` to reuse
prepared parquet files. If it is unset, the runner invokes VERL's original
data-preparation script into the run directory.

The proxy uses `load_hf_weights=False` for its synthetic checkpoint. It is a
protocol and integration smoke, not model-quality or accuracy evidence. In one
two-step random-weight proxy run, the observed rollout probability maximum
difference was `1.41e-05`, Pearson correlation was `0.5248`, rollout
correction/KL was `0.02708`, step time was `113.5s`, and peak memory was
`9.34 GiB` per GPU. Those figures are operational observations only and must
not be used to claim numerical accuracy.
