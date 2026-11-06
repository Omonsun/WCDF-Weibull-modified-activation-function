# WCDF Activation Function

[![Paper](https://img.shields.io/badge/Paper-IEEE-blue)](https://doi.org/10.1109/ICCIT68739.2025.11489541)
[![DOI](https://img.shields.io/badge/DOI-10.1109%2FICCIT68739.2025.11489541-blue)](https://doi.org/10.1109/ICCIT68739.2025.11489541)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Official TensorFlow/Keras implementation of the **Weibull Cumulative Distribution Function**, or **WCDF**, activation function.

WCDF is a smooth, bounded, and distribution-informed activation function introduced in the paper:

> **WCDF: A Modified Weibull Distribution-Based Deep Learning Activation Function**  
> Omon Das, Swaranjit Roy, and Mir Tawsif Sultan  
> 2025 28th International Conference on Computer and Information Technology (ICCIT)

**Paper:** [https://doi.org/10.1109/ICCIT68739.2025.11489541](https://doi.org/10.1109/ICCIT68739.2025.11489541)

## Abstract

Activation functions have a substantial influence on the optimization, stability, and generalization of deep neural networks. However, statistical distribution-based activation functions remain relatively underexplored.

WCDF uses the cumulative distribution function of the Weibull distribution as a hidden-layer activation function. It introduces two fixed hyperparameters, shape and scale, that control the curvature and sensitivity of the activation.

The function was evaluated against ReLU and GELU on MNIST, Fashion-MNIST, CIFAR-10, and CIFAR-100. The experiments showed lower test loss, stable convergence, and reduced train-validation gaps across the evaluated datasets.

## Mathematical Definition

For input \(x\), shape parameter \(k\), and scale parameter \(\lambda\), WCDF is defined as:

$\operatorname{WCDF}(x; k,\lambda) =\begin{cases}1-\exp\left[-\left(\frac{x}{\lambda}\right)^k\right],& x \geq 0, \\[6pt]0,& x < 0.\end{cases}$

where:

- $x$ is the input to the activation function
- $k > 0$ is the shape parameter
- $\lambda > 0$ is the scale parameter

The shape parameter controls how sharply the function rises, while the scale parameter controls the horizontal stretching or compression of the activation curve.

## Characteristics

WCDF provides the following properties:

- smooth nonlinear transformation for positive inputs
- bounded output in the interval \([0,1)\)
- fixed but tunable shape and scale hyperparameters
- task-specific curvature control
- non-zero gradients over its effective positive domain
- stable optimization behavior
- potential reduction of overfitting through bounded activations

The shape and scale values are fixed when the layer is created. They are not learned during model training.

## TensorFlow/Keras Implementation

```python
import tensorflow as tf
from tensorflow import keras


@keras.utils.register_keras_serializable(package="WCDF")
class Weibull(keras.layers.Layer):
    """
    Weibull CDF activation layer.

    Args:
        shape: Positive shape parameter k.
        scale: Positive scale parameter lambda.
        **kwargs: Additional arguments accepted by keras.layers.Layer.
    """

    def __init__(self, shape, scale, **kwargs):
        super().__init__(**kwargs)

        if shape <= 0:
            raise ValueError("`shape` must be greater than zero.")

        if scale <= 0:
            raise ValueError("`scale` must be greater than zero.")

        self.shape = float(shape)
        self.scale = float(scale)

    def call(self, inputs):
        inputs = tf.cast(inputs, tf.float32)

        # Restrict the Weibull CDF calculation to non-negative values.
        z = tf.maximum(inputs / self.scale, 0.0)

        # Weibull CDF: 1 - exp(-(x / scale)^shape)
        weibull_cdf = 1.0 - tf.exp(-tf.pow(z, self.shape))

        # The Weibull CDF is defined as zero for negative inputs here.
        return tf.where(inputs < 0.0, 0.0, weibull_cdf)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "shape": self.shape,
                "scale": self.scale,
            }
        )
        return config
```

The `register_keras_serializable` decorator allows models containing the custom layer to be saved and loaded without manually supplying the layer through `custom_objects`.

## Example Usage

```python
import tensorflow as tf
from tensorflow import keras

model = keras.Sequential(
    [
        keras.layers.Input(shape=(784,)),
        keras.layers.Dense(128),
        Weibull(shape=1.5, scale=1.0),
        keras.layers.Dense(64),
        Weibull(shape=1.5, scale=1.0),
        keras.layers.Dense(10, activation="softmax"),
    ]
)

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()
```

## CNN Example

```python
model = keras.Sequential(
    [
        keras.layers.Input(shape=(28, 28, 1)),
        keras.layers.Conv2D(
            filters=32,
            kernel_size=3,
            padding="same",
        ),
        keras.layers.BatchNormalization(),
        Weibull(shape=1.5, scale=1.0),
        keras.layers.MaxPooling2D(pool_size=2),
        keras.layers.Flatten(),
        keras.layers.Dense(10, activation="softmax"),
    ]
)
```

## Parameter Selection

The paper used fixed shape and scale values selected according to dataset characteristics.

| Dataset | Shape \(k\) | Scale \(\lambda\) |
|---|---:|---:|
| MNIST | 1.5 | 1.0 |
| Fashion-MNIST | 1.5 | 1.0 |
| CIFAR-10 | 1.0 | 5.0 |
| CIFAR-100 | 1.0 | 5.0 |

For the grayscale datasets, \(k=1.5\) and \(\lambda=1\) were used to provide a gradual rise while maintaining a normal sensitivity range.

For the color-image datasets, \(k=1\) and \(\lambda=5\) were used to broaden the sensitivity range for more complex and noisy inputs.

These values should be treated as experimentally evaluated starting points rather than universally optimal defaults.

## Parameter Interpretation

### Shape parameter

The shape parameter \(k\) controls the curvature and rate of increase:

- lower values produce a more gradual transition
- higher values produce a sharper transition
- changing \(k\) modifies the gradient profile of the function

### Scale parameter

The scale parameter \(\lambda\) controls horizontal sensitivity:

- smaller values compress the curve
- larger values stretch the curve
- larger values allow the activation to respond over a broader input range

Both parameters must be strictly greater than zero.

## Experimental Evaluation

WCDF was compared with ReLU and GELU using lightweight convolutional neural networks on four image-classification datasets:

- MNIST
- Fashion-MNIST
- CIFAR-10
- CIFAR-100

The experiments focused on test loss, classification accuracy, convergence behavior, and train-validation loss differences.

## Results

| Dataset | Activation | Epochs | Test loss | Test accuracy |
|---|---|---:|---:|---:|
| MNIST | ReLU | 150 | 0.068 | 0.980 |
| MNIST | GELU | 150 | 0.064 | 0.980 |
| MNIST | WCDF | 150 | **0.062** | **0.980** |
| Fashion-MNIST | ReLU | 150 | 0.290 | 0.901 |
| Fashion-MNIST | GELU | 150 | 0.310 | 0.896 |
| Fashion-MNIST | WCDF | 150 | **0.280** | **0.902** |
| CIFAR-10 | ReLU | 100 | 3.220 | 0.563 |
| CIFAR-10 | GELU | 100 | 2.460 | 0.562 |
| CIFAR-10 | WCDF | 100 | **1.120** | **0.624** |
| CIFAR-100 | ReLU | 100 | 3.784 | 0.281 |
| CIFAR-100 | GELU | 100 | 3.478 | 0.282 |
| CIFAR-100 | WCDF | 100 | **2.958** | **0.302** |

## Reported Test-Loss Improvements

| Dataset | Compared with ReLU | Compared with GELU |
|---|---:|---:|
| MNIST | 8.82% | 3.13% |
| Fashion-MNIST | 4.49% | 9.50% |
| CIFAR-10 | 65.26% | 54.47% |
| CIFAR-100 | 21.79% | 14.98% |

WCDF achieved the lowest reported test loss on all four evaluated datasets. The largest reduction was observed on CIFAR-10, where WCDF reduced test loss by 65.26% relative to ReLU and 54.47% relative to GELU.

These results apply to the architectures, parameter settings, data splits, and training configurations described in the paper. They should not be interpreted as a guarantee that WCDF will outperform other activation functions in every model or task.

## Saving and Loading a Model

The provided layer is registered as a Keras-serializable component.

```python
model.save("wcdf_model.keras")

restored_model = keras.models.load_model("wcdf_model.keras")
```

If the registration decorator is removed, load the model by explicitly passing the layer:

```python
restored_model = keras.models.load_model(
    "wcdf_model.keras",
    custom_objects={"Weibull": Weibull},
)
```

## Limitations

The current study has several limitations:

- shape and scale parameters were selected manually
- only a limited number of parameter combinations were evaluated
- evaluation was limited to image-classification benchmarks
- performance on deeper and more complex architectures remains to be investigated

Future research may evaluate WCDF in ResNet architectures, transformers, recurrent neural networks, object-detection systems, natural-language processing, and tabular-data models.

Another research direction is to investigate trainable or layer-specific shape and scale parameters. The implementation in this repository keeps both parameters fixed to remain consistent with the reported formulation.

## Repository Structure

A suggested repository structure is:

```text
wcdf-activation-function/
├── README.md
├── LICENSE
├── wcdf.py
└── examples/
    └── basic_usage.py
```

The core layer can be stored in `wcdf.py` and imported as follows:

```python
from wcdf import Weibull
```

## Reproducibility

For a reproducible comparison, keep the following elements consistent across activation functions:

- network architecture
- weight initialization
- optimizer and learning rate
- batch size
- number of epochs
- data split
- preprocessing procedure
- random seed

Only the activation function should change when comparing WCDF with ReLU, GELU, or another baseline.

## Citation

If you use WCDF in research, software, or experiments, please cite the paper using its DOI:

[https://doi.org/10.1109/ICCIT68739.2025.11489541](https://doi.org/10.1109/ICCIT68739.2025.11489541)

You may also include the DOI in a BibTeX entry:

```bibtex
@inproceedings{wcdf2025,
  title = {WCDF: A Modified Weibull Distribution-Based Deep Learning Activation Function},
  author = {Das, Omon and Roy, Swaranjit and Sultan, Mir Tawsif},
  booktitle = {2025 28th International Conference on Computer and Information Technology (ICCIT)},
  year = {2025},
  publisher = {IEEE},
  doi = {10.1109/ICCIT68739.2025.11489541},
  url = {https://doi.org/10.1109/ICCIT68739.2025.11489541}
}
```

## License

The code in this repository may be released under the MIT License. Add a `LICENSE` file containing the official MIT License text before publishing the repository.

The research paper remains subject to the copyright and usage terms of its publisher. The repository license applies only to the source code and repository materials for which the authors hold the necessary rights.

## Acknowledgment

If this implementation contributes to your research, please cite the original paper and link to this repository.
