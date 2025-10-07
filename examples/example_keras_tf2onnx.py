import numpy as np
import onnx
import tensorflow as tf
import tf2onnx
import tf2onnx.keras2onnx_api


class LeNet(tf.keras.Model):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv2d_1 = tf.keras.layers.Conv2D(filters=6,
                                               kernel_size=(3, 3), activation='relu',
                                               input_shape=(32, 32, 1))
        self.average_pool = tf.keras.layers.AveragePooling2D((3, 3))
        self.conv2d_2 = tf.keras.layers.Conv2D(filters=16,
                                               kernel_size=(3, 3), activation='relu')
        self.flatten = tf.keras.layers.Flatten()
        self.fc_1 = tf.keras.layers.Dense(120, activation='relu')
        self.fc_2 = tf.keras.layers.Dense(84, activation='relu')
        self.out = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, inputs, **kwargs):
        x = self.conv2d_1(inputs)
        x = self.average_pool(x)
        x = self.conv2d_2(x)
        x = self.average_pool(x)
        x = self.flatten(x)
        x = self.fc_2(self.fc_1(x))
        return self.out(x)


# Define a simple model
model = LeNet()
data = np.random.rand(2 * 416 * 416 * 3).astype(np.float32).reshape(2, 416, 416, 3)
expected = model(data)

# Get ConcreteFunction
# concrete_func = tf.function(model).get_concrete_function(tf.TensorSpec([None, None, None, None], tf.float32))
oxml = tf2onnx.keras2onnx_api.convert_keras(model, input_signature=[tf.TensorSpec([None, None, None, None], tf.float32)])
onnx.save(oxml, "model.onnx")
