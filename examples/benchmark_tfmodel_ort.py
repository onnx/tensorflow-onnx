"""
The following code compares the speed of tensorflow against onnxruntime
with a model downloaded from Tensorflow Hub.
"""
import time
import numpy
from tqdm import tqdm
import tensorflow_hub as hub
import tf2onnx
import onnxruntime as ort


def generate_random_images(shape=(100, 100), n=10):
    imgs = []
    for i in range(n):
        sh = (1,) + shape + (3,)
        img = numpy.clip(numpy.abs(numpy.random.randn(*sh)), 0, 1) * 255
        img = img.astype(numpy.float32)
        imgs.append(img)
    return imgs


def measure_time(fct, imgs):
    results = []
    times = []
    for img in tqdm(imgs):
        begin = time.perf_counter()
        result = fct(img)
        end = time.perf_counter()
        results.append(result)
        times.append(end - begin)
    return results, times


imgs = generate_random_images()

# Download model from https://tfhub.dev/captain-pool/esrgan-tf2/1
# python -m tf2onnx.convert --saved-model esrgan --output "esrgan-tf2.onnx" --opset 12
ort = ort.InferenceSession('esrgan-tf2.onnx')
fct_ort = lambda img: ort.run(None, {'input_0:0': img})
results_ort, duration_ort = measure_time(fct_ort, imgs)
print(len(imgs), duration_ort)

model = hub.load("https://tfhub.dev/captain-pool/esrgan-tf2/1")
results_tf, duration_tf = measure_time(model, imgs)
print(len(imgs), duration_tf)

print("ratio ORT / TF", sum(duration_ort) / sum(duration_tf))
