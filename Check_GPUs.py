import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # show all logs
import tensorflow as tf
from tensorflow.python.client import device_lib

# List all devices TensorFlow sees
devices = device_lib.list_local_devices()
print("All devices detected by TensorFlow:")
for d in devices:
    print(d)

# Specifically check GPUs
gpus = tf.config.list_physical_devices('GPU')
print("\nGPUs detected by TensorFlow:", gpus)
