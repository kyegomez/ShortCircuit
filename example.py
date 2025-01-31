import torch
from short_circuit_torch.main import ShortCircuitNet

# Create an instance of the ShortCircuitNet model with the specified parameters
model = ShortCircuitNet(512, 6, 8, 64, 2048, 0.1)

# Generate a random input tensor of shape (1, 512, 512)
input_tensor = torch.randn(1, 512, 512)

# Pass the input tensor through the model to get the output tensor
output_tensor = model(input_tensor)

# Print the output tensor
print(output_tensor)
