[![Multi-Modality](agorabanner.png)](https://discord.com/servers/agora-999382051935506503)


# ShortCircuit


## Install
```
pip3 install -U short-circuit-torch

```


## Example

```python
import torch 
from shortcircuit.main import ShortCircuitNet

# Create an instance of the ShortCircuitNet model with the specified parameters
model = ShortCircuitNet(512, 6, 8, 64, 2048, 0.1)

# Generate a random input tensor of shape (1, 512, 512)
input_tensor = torch.randn(1, 512, 512)

# Pass the input tensor through the model to get the output tensor
output_tensor = model(input_tensor)

# Print the output tensor
print(output_tensor)
```



# Missing
Input Sequence:
Node Hidden
Embeddings
Target Sequence:
Target Hidden
Embedding


# License
MIT
