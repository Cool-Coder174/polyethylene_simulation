import platform
import psutil
import subprocess
import sys
import pytest
from pathlib import Path

def get_system_info():
    """Gathers and returns system information."""
    info = {
        "os": platform.system(),
        "cpu_count": psutil.cpu_count(logical=True),
        "total_ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "gpu_info": "No NVIDIA GPU detected or py3nvml not installed."
    }

    if info["os"] == "Linux" or info["os"] == "Windows":
        try:
            # Check for NVIDIA GPU using nvidia-smi
            # This is a more robust check than relying solely on py3nvml
            # as py3nvml might be installed but no GPU present.
            result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
                                    capture_output=True, text=True, check=True)
            gpu_lines = result.stdout.strip().split('\n')
            if gpu_lines and gpu_lines[0]:
                info["gpu_info"] = {"present": True, "details": [line.strip() for line in gpu_lines]}
            else:
                info["gpu_info"] = {"present": False, "details": []}
        except (subprocess.CalledProcessError, FileNotFoundError):
            info["gpu_info"] = {"present": False, "details": []}
    elif info["os"] == "Darwin": # macOS
        # For macOS, checking for Metal/OpenCL capabilities is more complex
        # and often requires specific frameworks. For simplicity, we'll assume
        # no dedicated GPU acceleration for now unless explicitly configured.
        info["gpu_info"] = {"present": False, "details": []}

    return info

def run_tests_conditionally():
    """Runs tests based on detected system specifications."""
    system_info = get_system_info()
    print("\n--- System Information ---")
    for key, value in system_info.items():
        print(f"{key.replace('_', ' ').title()}: {value}")
    print("--------------------------\n")

    test_paths = []
    # Always include basic unit tests
    test_paths.append(str(Path("tests/")))

    # Conditionally include GPU tests
    if isinstance(system_info["gpu_info"], dict) and system_info["gpu_info"].get("present", False):
        print("NVIDIA GPU detected. Including GPU acceleration tests.")
        test_paths.append(str(Path("hw_accel_cuda/")))
    else:
        print("No NVIDIA GPU detected or not on Linux/Windows. Skipping GPU acceleration tests.")

    print(f"Running tests from: {test_paths}")
    # Run pytest with the selected test paths
    # We use sys.exit(pytest.main(...)) to ensure the script exits with the correct status code
    sys.exit(pytest.main(test_paths))

if __name__ == "__main__":
    run_tests_conditionally()
