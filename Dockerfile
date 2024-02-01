# Use an official Ubuntu as a parent image
FROM ubuntu:latest

# Set environment variables to avoid user interaction during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install Python and pip
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip to the latest version
RUN pip3 install --upgrade pip

# Install TensorFlow (CPU version)
RUN pip3 install tensorflow

# Optional: Install additional common Python data science libraries
RUN pip3 install numpy pandas matplotlib ipython

# Install PyTorch
# Note: The specific PyTorch install command depends on your system (CPU/GPU, CUDA version).
# The following command installs the CPU version of PyTorch.
RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Set the working directory in the container
WORKDIR /workspace

# Command to run when starting the container
CMD ["bash"]

