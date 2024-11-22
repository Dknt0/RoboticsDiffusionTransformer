RDT Note
===

# Network Information

Input: Language instruction. RGB images of up to 3 views.

Output: Next 64 robot actions. Joint position or velocity, EEF.

Single-arm, dual-arm.

Parameter: 1B parameters. Or a small version with 170M parameters.

# 1 Fine-Tuning

Clone the repository. Clone the two encoders `t5-v1_1-xxl` and `siglip`. Note that the model of the two encoders are large, and we need to use `git lfs` to install them. After that, create symbolic links to encoders.

Fill the missing arguments in the `config/base.yaml` file.

## 1.1 Prepare the Dataset

> Refer to this [tutorial](https://github.com/thu-ml/RoboticsDiffusionTransformer?tab=readme-ov-file#fine-tuning-on-your-own-dataset).

Link the dataset here:

```shell
ln -s /path/to/my_cool_dataset datasets/my_cool_dataset
```

Then we should implement the dataset loader.

1. Register the configuration. There are three files to be modified: `configs/dataset_control_freq.json`, `configs/finetune_datasets.json` and `configs/finetune_sample_weights.json`. Replace the placeholder `agilex` with our dataset name.
2. Re-implement `HDF5VLADataset` in `data/hdf5_vla_dataset.py`.
  * Fill the robot action into the unified action vector. Refer to `configs/state_vec.py`.
  * 6D representation is used for EEF orientation. See this [paper](https://arxiv.org/pdf/1812.07035) and [script](docs/test_6drot.py).
  * No physical quantities are normalized during pre-training, except the gripper width.
  * We need to precompute the language embeddings. See `scripts/encode_lang_batch.py`.
3. Run this command to compute statistics information for the dataset.

```shell
python -m data.compute_dataset_stat_hdf5
```

Configuration relevant to model are in `config/base.yaml`.

We need to change some parameters in `finetune.sh`, such as `CUTLASS_PATH` and `WANDB_PROJECT`. Run this command to start fine-tuning:

```shell
source finetune.sh
```

> `Cutllass` is a library for high-performance matrix multiplication provided by NVIDIA. It contains only header files. We can simply clone its git repository then add the path to `CUTLASS_PATH`.

> `Wandb` is a tool for online experiment tracking.

When run the script, check following tips:

1. Specify `--precomp_lang_embed` in the `finetune.sh` to use the precomputed language embeddings.
2. `pretrained_model_name_or_path` should be filled as one of following options:
  * Path to the pre-trained model downloaded from HuggingFace.
  * Path to a model weights saved in `checkpoints`
  * Path to a model weights saved by DeepSpeed
  * `None` to randomly initialize the model.
3. We can use TensorBoard to monitor the training process. Fine-tuning is over when the `overall_avg_sample_msg` converges.
4. If the training oscillates, we can increase the batch size by adding more GPUs or setting a larger `--gradient_accumulation_steps`

# 2 Deployment

A class named `RoboticDiffusionTransformerModel` is implemented in `scripts/agilex_model.py`. We can call `step()` of that class for inference.

An example hardware for deployment on Mobile ALOHA is provided in `scripts/agilex_inference.py`. The corresponding running script is in `inference.sh`.

Our GPU memory is not enough to encode the language. Refer to `scripts/encode_lang.py` for precomputation. For VRAM less than 24GB, enable offloading by specifying an offload directory.

# Plan

* 2024.11.12

Study some basis.

* 2023.11.13

Study another basis.

Try pre-encode. It works well.

* 2024.11.14

Try to record some HDF5 samples in C++.

* 2024.11.15

Finish the c++ recorder class. Finished.

* 2024.11.18

Test the fine-tuning code on a single 4080 Super. Passed.

---

Try the `encode_lang_batch.py` and `encode_lang.py` scripts. <text style="color:#00FF00; background-color:#FFFF0033;">Done &#128077;</text>

What the hell is hdf5 dataset? <text style="color:#00FF00; background-color:#FFFF0033;">Done &#128077;</text>

Try fine-tuning code. <text style="color:#00FF00; background-color:#FFFF0033;">Done &#128077;</text>

Try to run the model on. Run the inference code.

Import the Mobile ALOHA in simulation. Gazebo, Isaac Sim or the other simulator. I prefer the middle one.

* TODO

How to make a fine-turning dataset?

Read the source code.

Deployment.

**Bug**: The pre-computed by `encode_lang_batch.py` embeddings are not used properly. They have the name like `lang_embeds_0.pt`. But the script `main.py` only loads the `lang_embeds.pt`.

# Idea

Use 1B-model or 170M-model?

Run the model in a simulation? We should have a dual-arm robot with a controller.

See the `configs/state_vec.py`, the definition of gripper joint. There 5 DoF. Can we use these DoF for our hand?

```py
**{
    'gripper_joint_{}_pos'.format(i): i + 10 for i in range(5)
  }
```

