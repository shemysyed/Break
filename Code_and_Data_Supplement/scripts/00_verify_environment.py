import platform, sys
import numpy, pandas, PIL, sklearn, torch, torchvision, transformers

print("Operating system:", platform.platform())
print("Python:", sys.version.split()[0])
print("PyTorch:", torch.__version__)
print("CUDA used by PyTorch:", torch.version.cuda)
print("cuDNN:", torch.backends.cudnn.version())
print("Torchvision:", torchvision.__version__)
print("Transformers:", transformers.__version__)
print("scikit-learn:", sklearn.__version__)
print("NumPy:", numpy.__version__)
print("Pandas:", pandas.__version__)
print("Pillow:", PIL.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
