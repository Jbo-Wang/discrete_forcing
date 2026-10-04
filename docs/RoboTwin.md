# RoboTwin Clean 50

This directory provides RoboTwin clean-task training and evaluation with the
Discrete Forcing model. See the configuration for model and training details.

## Environment

Install the training environment as in the [LIBERO guide](LIBERO.md). Install
RoboTwin separately for simulation; its Python environment is used only for
evaluation.

## Data

Place the 50 LeRobot-format clean datasets under `datasets/robotwin/Clean/`.
Each task needs `meta/modality.json`. If it is absent, use the supplied
[template](../examples/Robotwin/train_files/modality.json).

```bash
for task in datasets/robotwin/Clean/*; do
  test -e "$task/meta/modality.json" ||
    cp examples/Robotwin/train_files/modality.json "$task/meta/modality.json"
done
```

## Training

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 MAIN_PROCESS_PORT=29701 \
  bash train_robotwin.sh \
  framework.qwenvl.base_vlm=/path/to/Qwen3.5-4B \
  datasets.vla_data.data_root_dir=/path/to/robotwin-datasets
```

Adjust the [configuration](../examples/Robotwin/train_files/robotwin_clean_method.yaml)
for your hardware and data paths. Use a new `run_id` for each run.

## Evaluation

```bash
export ROBOTWIN_HOME=/path/to/RoboTwin
export ROBOTWIN_PYTHON=/path/to/robotwin-env/bin/python
export STARVLA_PYTHON=/path/to/train-env/bin/python
bash eval_robotwin.sh \
  --checkpoint results/robotwin_clean_method_4b/checkpoints/steps_60000_pytorch_model.pt \
  --gpus 0,1,2,3,4,5,6,7 --port-base 5694
```

This runs `demo_clean` for all 50 tasks with 100 rollouts per task. The wrapper
starts the policy server, uses the checkpoint's normalization statistics, and
writes per-task logs plus `summary.tsv` under `robotwin_eval_logs/`. Use
`--task TASK` to evaluate one task. Keep the checkpoint in its run's
`checkpoints/` directory alongside the parent run's `config.yaml` and
`dataset_statistics.json`.
