Tips to convert TensorFlow RNN models.
========

# Status

Supported and tested RNN classes and APIs: LSTMCell, BasicLSTMCell, GRUCell, GRUBlockCell, MultiRNNCell, and user defined RNN cells inheriting from rnn_cell_impl.RNNCell, used along with DropoutWrapper, BahdanauAttention, AttentionWrapper. Check [here](tests/test_custom_rnncell.py) for well tested cases.

For other advanced RNN cells, it is supposed to good to convert as well, but there is no comprehensive testing against them.

```tf.nn.dynamic_rnn``` and ```tf.nn.bidirectional_dynamic_rnn``` are common APIs to trigger RNN cell's run, both approaches are supported to convert.

# Commands

Use following commands to have a quick trial on your model:

```
python -m tf2onnx.convert --input frozen_rnn_model.pb --inputs input1:0,input2:0 --outputs output1:0,output2:0 --fold_const --opset 8 --output target.onnx  --continue_on_error
```

## Limitation

Besides BasicLSTMCell/LSTMCell/GRUCell/GRUBlockCell conversion, all other conversion requires target onnx opset to be >= 8.

## Samples

There are a few tests case of [LSTMCell, BasicLSTMCell](tests/test_lstm.py), [GRUCell](tests/test_gru.py), [GRUBlockCell](tests/test_grublock.py) for your reference.

For other advanced RNN cells, check [here](tests/test_custom_rnncell.py).

# Verify Correctness

Use [onnxruntime](https://github.com/Microsoft/onnxruntime) or [caffe2](https://caffe2.ai/) to test against converted models.

There is a simpler way to run your models and test its correctness (compared with TensorFlow run) using following command.

```
python tests\run_pretrained_models.py --backend onnxruntime  --config rnn.yaml --tests model_name --fold_const --onnx-file ".\tmp" --opset 8
```

The content of rnn.yaml looks as below. For inputs, an explicit numpy expression or a shape can be used. If a shape is specified, the value will be randomly generated.

```
model_name:
  model: path/to/tf_frozen.pb
  input_get: get_ramp
  inputs:
    "input1:0": np.array([60])  # numpy random function
    "input2:0": [2, 1, 300]  # shape for the input
  outputs:
    - output1:0
    - output2:0
```
