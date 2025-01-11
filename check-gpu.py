import torch
import tensorflow as tf

# Check available GPUs in PyTorch
def check_pytorch_gpus():
    pytorch_gpus = torch.cuda.device_count()
    print(f"PyTorch detected {pytorch_gpus} GPU(s).")
    for i in range(pytorch_gpus):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

# Check available GPUs in TensorFlow
def check_tensorflow_gpus():
    tensorflow_gpus = len(tf.config.experimental.list_physical_devices('GPU'))
    print(f"TensorFlow detected {tensorflow_gpus} GPU(s).")
    for i in range(tensorflow_gpus):
        print(f"GPU {i}: {tf.config.experimental.list_physical_devices('GPU')[i]}")

# Check for general GPU information
def check_gpu_info():
    try:
        from subprocess import check_output
        print("\nGeneral GPU Information:")
        gpu_info = check_output(["nvidia-smi", "--query-gpu=name,memory.total,memory.free,memory.used", "--format=csv,noheader,nounits"])
        print(gpu_info.decode('utf-8'))
    except Exception as e:
        print("Unable to retrieve general GPU info. Ensure that NVIDIA drivers and `nvidia-smi` are installed.")

if __name__ == '__main__':
    check_pytorch_gpus()
    check_tensorflow_gpus()
    check_gpu_info()
