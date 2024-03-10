# Pan-Mamba

This folder contains the implementation of the Pan-Mamba model for pan-sharpening. 

**We have released pre-trained checkpoints for Pan-Mamba, trained on the WV2, WV3, and Gaofen-2 satellites. The checkpoints trained on 4 channels (RGB and Near-Infrared) images. Please note that the training datasets are not currently public. You are encouraged to test the checkpoints on your own satellite datasets.**

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

In order to run this implementation, you need to have the following software and libraries installed:

- Python 3.7 or later
- PyTorch 1.6 or later
- CUDA (if using GPU)
- NumPy
- Matplotlib
- OpenCV
- PyYAML

### Installing
**Please follow the https://github.com/alexhe101/Pan-Mamba/blob/main/Mamba-block/readme.md to install mamba block first.**

You can install the necessary packages using pip:

```python
pip install torch numpy matplotlib opencv-python pyyaml
```

### Configuration

Before training the model, you need to configure the following options in the `option.yaml` file:

- `log_dir`: the directory to store the training log files.
- `checkpoint`: the directory to store the trained model parameters.
- `data_dir_train`: the directory of the training data.
- `data_dir_eval`: the directory of the evaluation data.

### Training the Model

To train the model, you can run the following command:

```
python main.py
```

### Testing the Model

To test the trained MSDDN model, you can run the following command:

```
python test.py
python py-tra/demo_deep_methods.py
```

### Configuration

The configuration options are stored in the `option.yaml` file. Here is an explanation of each of the options:

#### algorithm

- algorithm: The model for training

#### Logging

- `log_dir`: The location where the log files will be stored.

#### Model Weights

- `checkpoint`: The location to store the model weights.

#### Training Data

- `data_dir_train`: The location of the training data.
- `data_dir_eval`: The location of the test data.

#### Pretrain

- `pretrained`: Whether to use a pretrained model.
- `pre_sr`: The location of the pretrained model.
- `pre_folder`: The location where the pretrained models are stored.

#### Testing

- `algorithm`: The algorithm to use for testing.
- `type`: The type of testing, either `test` or `eval`.
- `data_dir`: The location of the test data.
- `source_ms`: The source of the multi-spectral data.
- `source_pan`: The source of the panchromatic data.
- `model`: The location of the model to use for testing.
- `save_dir`: The location to save the test results.

#### Data Processing

- `upscale`: The upscale factor.
- `batch_size`: The size of each batch.
- `patch_size`: The size of each patch.
- `data_augmentation`: Whether to use data augmentation.
- `n_colors`: The number of color channels.
- `rgb_range`: The range of the RGB values.
- `normalize`: Whether to normalize the data.

#### Training Hyperparameters

- `schedule.lr`: The learning rate.
- `schedule.decay`: The learning rate decay.
- `schedule.gamma`: The learning rate decay factor.
- `schedule.optimizer`: The optimizer to use, either `ADAM`, `SGD`, or `RMSprop`.
- `schedule.momentum`: The momentum for the `SGD` optimizer.
