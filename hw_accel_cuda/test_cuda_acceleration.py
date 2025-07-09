import time
import numpy as np
import psutil
import pytest
from py3nvml import py3nvml

# Try to initialize NVML and skip all tests in this module if it fails
try:
    py3nvml.nvmlInit()
    NVML_AVAILABLE = True
except py3nvml.NVMLError_LibraryNotFound:
    NVML_AVAILABLE = False

pytest.mark.skipif(not NVML_AVAILABLE, reason="NVML library not found, skipping CUDA tests.")

# Try to import Numba and get GPU handle
try:
    from numba import cuda
    gpu_handle = py3nvml.nvmlDeviceGetHandleByIndex(0)
    NUMBA_AVAILABLE = True
except (ImportError, py3nvml.NVMLError):
    NUMBA_AVAILABLE = False

# Simple CUDA kernel: elementwise square
@cuda.jit
def square_kernel(input_arr, output_arr):
    idx = cuda.grid(1)
    if idx < input_arr.size:
        output_arr[idx] = input_arr[idx] ** 2

def measure_gpu_utilization():
    util = py3nvml.nvmlDeviceGetUtilizationRates(gpu_handle)
    return util.gpu, util.memory  # percentages

def measure_cpu_ram():
    cpu = psutil.cpu_percent(interval=0.5)
    mem = psutil.virtual_memory().percent
    return cpu, mem

@pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Numba not available")
@pytest.mark.parametrize("size", [1<<20, 1<<22])
def test_cuda_kernel_correctness_and_performance(size):
    # Prepare data
    h_in = np.arange(size, dtype=np.float32)
    h_out = np.zeros_like(h_in)
    
    # Transfer to GPU
    d_in = cuda.to_device(h_in)
    d_out = cuda.to_device(h_out)
    
    # Launch kernel
    threads_per_block = 256
    blocks = (size + threads_per_block - 1) // threads_per_block

    # Warm-up
    square_kernel[blocks, threads_per_block](d_in, d_out)
    cuda.synchronize()
    
    # Measure GPU usage before timed run
    gpu_before, mem_before = measure_gpu_utilization()
    cpu_before, ram_before = measure_cpu_ram()
    
    # Timed GPU execution
    start_gpu = time.time()
    square_kernel[blocks, threads_per_block](d_in, d_out)
    cuda.synchronize()
    gpu_time = time.time() - start_gpu

    # Fetch result
    result = d_out.copy_to_host()
    assert np.allclose(result, h_in ** 2), "CUDA kernel produced incorrect results"

    # Measure utilization after run
    gpu_after, mem_after = measure_gpu_utilization()
    cpu_after, ram_after = measure_cpu_ram()

    # Define thresholds for resource usage jump
    gpu_threshold = 10
    mem_threshold = 5
    cpu_threshold = 5
    ram_threshold = 1

    # Validate resource usage jump
    assert gpu_after >= max(gpu_before + gpu_threshold, 30), f"Low GPU usage: {gpu_after}%"
    assert mem_after >= max(mem_before + mem_threshold, 20), f"Low GPU memory usage: {mem_after}%"
    assert cpu_after >= max(cpu_before + cpu_threshold, 20), f"Low CPU usage: {cpu_after}%"
    assert ram_after >= max(ram_before + ram_threshold, 20), f"Low RAM usage: {ram_after}%"

    # Compare to CPU-only run
    start_cpu = time.time()
    cpu_out = h_in ** 2
    cpu_time = time.time() - start_cpu

    assert gpu_time * 2 < cpu_time, (
        f"GPU speedup insufficient: GPU {gpu_time:.3f}s vs CPU {cpu_time:.3f}s"
    )