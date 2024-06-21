# test_cuda.py
import torch

# PyTorch
print("PyTorch CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Current CUDA Device:", torch.cuda.current_device())
    print("Device Name:", torch.cuda.get_device_name(torch.cuda.current_device()))

