# metaworld-algorithms
Implementations of Multi-Task and Meta-Learning baselines for the Metaworld benchmark

## Installation

### From a clone of the repository

0. Install [uv](https://docs.astral.sh/uv/)
1. Create a virtual environment for the project (with Python>=3.12)
   ```
   uv venv .venv --python 3.12
   ```
3. Activate the virtual environment
   ```
   source .venv/bin/activate
   ```
5. Install the dependencies
   ```
   uv pip install -e ".[cuda12]"
   ```

> [!Important]
> Installing this package with no extras specified, i.e. `uv pip install -e .` **will not work**.
> You need to be explicit about your choice of accelerator by specifying the correct extra when installing.

> [!NOTE]
> The command in step 3. will install with NVIDIA GPU support. To use other accelerators, replace `cuda12` with the appropriate accelerator name.
> Valid options are:
> - `cpu` (No accelerator)
> - `tpu` (GCP TPUs)
> - `cuda12` (NVIDIA GPUs)
> - `metal` (Apple Silicon)
>
> For example, to install with TPU support, the proper commmand would be
> ```
> uv pip install -e ".[tpu]"
> ```


## Structure

Here is how you can navigate this repository:

- `examples` contains code for running baselines.
- `metaworld_algorithms/rl/algorithms` contains the implementations of baseline *algorithms* (e.g. MTSAC, MTPPO, MAML, etc).
- `metaworld_algorithms/nn` contains the implementations of *neural network architectures* used in multi-task RL (e.g. Soft-Modules, PaCo, MOORE, etc).
- `metaworld_algorithms/rl/networks.py` contains code that wraps these neural network building blocks into agent components (actor networks, critic networks, etc).
- `metaworld_algorithms/rl/buffers.py` contains code for the buffers used.
- `metaworld_algorithms/rl/algorithms/base.py` contains code for training loops (e.g. on-policy, off-policy, meta-rl).
- `meatworld_algorithms/envsmetaworld.py` contains utilities for wrapping metaworld for use with these baselines.

