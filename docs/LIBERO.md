# LIBERO Training and Evaluation

This repository provides the model, configuration, data loader, and evaluation code
for the LIBERO experiments. Training data and model weights are downloaded
separately. Run all commands from the repository root on Linux.

## Environment

Use separate Python 3.10 environments for training/policy serving and simulation.
A CUDA-compatible PyTorch installation and CUDA build toolkit are required for training.
Building CUDA extensions from source requires a C++17 compiler compatible with
the installed PyTorch headers.
Evaluation requires an NVIDIA GPU and EGL/OpenGL system libraries for offscreen
rendering. The evaluation wrapper also requires `ss`, `flock`, and `realpath`.

```bash
conda create -n libero_train python=3.10 -y
conda activate libero_train
python -m pip install pip==23.0.1 setuptools==80.9.0 wheel ninja packaging
pip install -r requirements-train.txt
pip install --no-build-isolation flash-attn==2.7.4.post1 causal-conv1d==1.6.1
pip install --no-deps -e .

conda create -n libero_eval python=3.10 -y
conda activate libero_eval
python -m pip install pip==23.0.1 setuptools==80.9.0 wheel
pip install -r requirements-eval.txt
# Preserve NumPy 1.26.4 for the simulator. OpenCV 4.12 declares a NumPy >=2
# requirement; --no-deps intentionally overrides that dependency declaration.
pip install --no-deps opencv-python==4.12.0.88
git clone https://github.com/Lifelong-Robot-Learning/LIBERO third_party/LIBERO
git -C third_party/LIBERO checkout 8f1084e3132a39270c3a13ebe37270a43ece2a01
pip install --no-deps -e third_party/LIBERO
python prepare_libero.py
export LIBERO_CONFIG_PATH="$PWD/third_party/LIBERO/libero"
export PYTHONPATH="$PWD:$PWD/third_party/LIBERO:${PYTHONPATH:-}"
```

`prepare_libero.py` generates machine-local simulator paths at installation time
and refuses to overwrite an existing configuration.

## Data

In the training environment:

```bash
bash prepare_data.sh
```

This downloads the four no-noops LIBERO LeRobot datasets into `datasets/` and installs
the provided modality description. Existing datasets can instead be selected using
`datasets.vla_data.data_root_dir=/path/to/datasets` when launching training. Each suite
must be a direct subdirectory with the names used in `prepare_data.sh`.

The VLM is `Qwen/Qwen3.5-0.8B`; it is downloaded by Transformers, or can be supplied
as a local directory using `framework.qwenvl.base_vlm=/path/to/model`.

## Training

```bash
conda activate libero_train
CUDA_VISIBLE_DEVICES=0,1,2,3 MAIN_PROCESS_PORT=29701 bash train.sh
```

The supplied config trains all four suites jointly for 80,000 optimizer steps with
four GPUs, batch size 16 per GPU, gradient accumulation 1, and diffusion repeat 8.
The action expert uses hidden dimension 1024, 2 shared layers and 22 layers per branch.
The VLM receives the primary and wrist images with the task instruction; robot
state is not injected into the model. Both action horizons are 8, with 7 action
dimensions and 256 bins per dimension in the discrete branch.

The continuous source is `0.3 * decode(GT bins) + 0.7 * Gaussian noise`.
Continuous-phase bin corruption retains approximately 30% of the GT bins and
replaces the others with different valid bins.

Configuration is in `examples/LIBERO/train_files/libero_method.yaml`. The total
loss is `Lc + 0.1 * Ldis`; `Lc` includes the configured auxiliary one-step loss.
Outputs are generated under `results/libero_method/`, including the configuration
and normalization statistics needed for evaluation. Use a new `run_id` for each
fresh run, for example `bash train.sh run_id=libero_method_run2`.

## Evaluation

Set each Python executable to the appropriate installed environment. The four
suites run sequentially on one GPU with 50 rollouts per task. Default inference
uses one discrete step followed by one continuous step. The client executes all
8 actions before replanning.

```bash
export STARVLA_PYTHON="$(conda run -n libero_train python -c 'import sys; print(sys.executable)')"
export LIBERO_PYTHON="$(conda run -n libero_eval python -c 'import sys; print(sys.executable)')"
bash examples/LIBERO/eval_files/eval_four_suites.sh \
  --checkpoint results/libero_method/checkpoints/steps_80000_pytorch_model.pt \
  --gpu 0 --port 7133 --trials 50
```

The wrapper starts the policy server, evaluates the suites, and aggregates the
generated results. Its default output directory is unique per invocation.
Each suite has 10 tasks. Its success rate is successful rollouts divided by total
rollouts; the final macro average is the unweighted mean of the four suite rates.
The summary also reports the pooled micro average. With 50 rollouts per task in
all suites, these two averages are equal. Logs and videos are generated only when
evaluation runs and are not included in this repository. A checkpoint must remain
inside its run's `checkpoints/` directory, with `config.yaml` and
`dataset_statistics.json` in the parent run directory.

## Checks

```bash
conda activate libero_train
python check_install.py
# Requires model weights, prepared training data, and a GPU:
python check_install.py --initialize

conda activate libero_eval
export LIBERO_CONFIG_PATH="$PWD/third_party/LIBERO/libero"
export PYTHONPATH="$PWD:$PWD/third_party/LIBERO:${PYTHONPATH:-}"
python check_install.py --evaluation
```

The initialization check creates a temporary output directory, loads the model,
builds the training dataloader, and reads one batch without starting training.
The evaluation check verifies imports; it does not execute evaluation rollouts.

## Third-Party Attribution

Upstream copyright notices, public contributor attributions, and license terms
are retained in the source files. LIBERO is installed separately from its public
repository at the revision specified above.
