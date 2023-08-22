# 🔱 TanhRelu Hybrid Activation (TraHA)

## 🚀 Introduction

Activation functions, typically denoted as $`f(x)`$, are pivotal in neural networks. They inject non-linearity between layers, empowering the network to unravel intricate patterns in the data. `tanh` and `ReLU` have historically been mainstream choices, each boasting its set of pros and cons. **TanhRelu Hybrid Activation (TraHA)** seamlessly merges the virtues of both functions and presents itself as a promising candidate, especially for architectures like the GRU.

## 📐 Mathematical Definition

For an input $`x`$ into the activation function, TraHA is defined as:

$`[f(x) = \alpha \times \tanh(x) + \beta \times \max(0, x)]`$

Where:
- $`\alpha`$ and $`\beta`$ are scalar hyperparameters.
- $`\tanh`$ is the hyperbolic tangent function, bounding outputs within $`([-1, 1])`$.
- $`\max(0, x)`$ signifies the ReLU component, introducing non-linearity for positive inputs.

## 💡 Rationale Behind TraHA

### 1. **Bounded & Zero-centered Output** 🔒:
The $`\tanh`$ component of TraHA ensures outputs are bounded and zero-centered. This is particularly advantageous for recurrent architectures like GRUs, mitigating potential instabilities.

### 2. **Non-saturating Region for Positive Inputs** 🌌:
With its `ReLU` segment, TraHA promises non-saturation for $`( x > 0 )`$, confronting the vanishing gradient challenge often seen in deep networks.

### 3. **Parameterized Flexibility** 🎛:
The dials $`\alpha`$ and $`\beta`$ facilitate fine-tuning, striking a balance between the `tanh` and `ReLU` dynamics tailored for specific tasks.

## 🌐 Potential Use Cases

- **Deep Neural Networks (DNNs)** 🌲: An ideal fit for deeper architectures, nourishing gradient flow.

- **Gated Recurrent Units (GRUs)** 🔁: By replacing the traditional `tanh` in GRUs with TraHA, one might achieve better convergence and stability, especially in long sequences.

- **Generative Models** 🎨: A harmonious balance between robust gradient dynamics and bounded outputs ensures stable epochs.

## 🖋 Conclusion

TanhRelu Hybrid Activation (TraHA) offers a refreshing perspective in the realm of neural activations. Blending the strengths of two dominant functions, its application, especially in GRUs, could be a game-changer. While its theoretical pillars are robust, the ultimate validation will be its empirical performance across diverse datasets and tasks.

## Usage with TensorFlow/Keras 🔥:
```python
import tensorflow as tf
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.layers import Activation
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

get_custom_objects().update({'traha': Activation(traha)})
model = Sequential([
    Dense(128, input_shape=(784,), activation='traha'),
    Dense(10, activation='softmax')
])
```

## Usage with PyTorch 🔥:
```python
import torch
import torch.nn as nn

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.traha = traha(alpha=1.0, beta=1.0)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.traha(self.fc1(x))
        x = nn.Softmax(dim=1)(self.fc2(x))
        return x

