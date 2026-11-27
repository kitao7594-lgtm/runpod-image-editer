# Base image ships a matched set: torch 2.8.0 + torchvision 0.23.0 +
# CUDA 12.8.1 userland (Blackwell-ready). Do NOT reinstall torch via pip:
# mixing pip torch (cu128) with the old CUDA 12.4 base caused an NVML
# assert (NVML_SUCCESS == r) inside CUDACachingAllocator on some hosts.
FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY handler.py .
CMD ["python", "-u", "handler.py"]
