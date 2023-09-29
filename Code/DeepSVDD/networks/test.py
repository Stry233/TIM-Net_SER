import torch
import torch.nn as nn

# Define the input dimensions
batch_size = 64
in_channels = 10
out_channels = 20
seq_length = 39

# Create a sample input tensor
input_data = torch.randn(batch_size, in_channels, seq_length)

# Define the 1D convolutional layer
conv1d_layer = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=2, padding=1)

# Perform the forward pass
output_data = conv1d_layer(input_data)

# Print the shape of the output
print(output_data.shape)  # This should print torch.Size([64, 20, 39])