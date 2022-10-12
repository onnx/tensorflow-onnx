# SPDX-License-Identifier: Apache-2.0

import os
import sys
import unittest
import mock_keras2onnx
import numpy as np
from mock_keras2onnx.proto import keras, is_tensorflow_older_than
from os.path import dirname, abspath
sys.path.insert(0, os.path.join(dirname(abspath(__file__)), '../../keras2onnx_tests/'))
from test_utils import run_onnx_runtime
from onnxconverter_common.onnx_ex import get_maximum_opset_supported

Activation = keras.layers.Activation
BatchNormalization = keras.layers.BatchNormalization
Bidirectional = keras.layers.Bidirectional
concatenate = keras.layers.concatenate
Conv1D = keras.layers.Conv1D
Dense = keras.layers.Dense
Dropout = keras.layers.Dropout
Embedding = keras.layers.Embedding
Flatten = keras.layers.Flatten
Input = keras.layers.Input
LeakyReLU = keras.layers.LeakyReLU
LSTM = keras.layers.LSTM
MaxPooling1D = keras.layers.MaxPooling1D
multiply = keras.layers.multiply
Reshape = keras.layers.Reshape
UpSampling2D = keras.layers.UpSampling2D

Sequential = keras.models.Sequential
Model = keras.models.Model


class TestNLP(unittest.TestCase):

    def setUp(self):
        self.model_files = []

    def tearDown(self):
        for fl in self.model_files:
            os.remove(fl)

    def test_addition_rnn(self):
        # An implementation of sequence to sequence learning for performing addition
        # from https://github.com/keras-team/keras/blob/master/examples/addition_rnn.py
        DIGITS = 3
        MAXLEN = DIGITS + 1 + DIGITS
        HIDDEN_SIZE = 128
        BATCH_SIZE = 128
        CHARS_LENGTH = 12

        for RNN in [keras.layers.LSTM, keras.layers.GRU, keras.layers.SimpleRNN]:
            model = keras.models.Sequential()
            model.add(RNN(HIDDEN_SIZE, input_shape=(MAXLEN, CHARS_LENGTH)))
            model.add(keras.layers.RepeatVector(DIGITS + 1))
            model.add(RNN(HIDDEN_SIZE, return_sequences=True))
            model.add(keras.layers.TimeDistributed(keras.layers.Dense(CHARS_LENGTH, activation='softmax')))
            onnx_model = mock_keras2onnx.convert_keras(model, model.name)
            x = np.random.rand(BATCH_SIZE, MAXLEN, CHARS_LENGTH).astype(np.float32)
            expected = model.predict(x)
            self.assertTrue(run_onnx_runtime(onnx_model.graph.name, onnx_model, x, expected, self.model_files))

    def test_babi_rnn(self):
        # two recurrent neural networks based upon a story and a question.
        # from https://github.com/keras-team/keras/blob/master/examples/babi_rnn.py
        RNN = keras.layers.recurrent.LSTM
        EMBED_HIDDEN_SIZE = 50
        SENT_HIDDEN_SIZE = 100
        QUERY_HIDDEN_SIZE = 100
        BATCH_SIZE = 32
        story_maxlen = 15
        vocab_size = 27
        query_maxlen = 17

        sentence = Input(shape=(story_maxlen,), dtype='int32')
        encoded_sentence = Embedding(vocab_size, EMBED_HIDDEN_SIZE)(sentence)
        encoded_sentence = RNN(SENT_HIDDEN_SIZE)(encoded_sentence)

        question = Input(shape=(query_maxlen,), dtype='int32')
        encoded_question = Embedding(vocab_size, EMBED_HIDDEN_SIZE)(question)
        encoded_question = RNN(QUERY_HIDDEN_SIZE)(encoded_question)

        merged = concatenate([encoded_sentence, encoded_question])
        preds = Dense(vocab_size, activation='softmax')(merged)

        model = Model([sentence, question], preds)
        onnx_model = mock_keras2onnx.convert_keras(model, model.name)
        x = np.random.randint(5, 10, size=(BATCH_SIZE, story_maxlen)).astype(np.int32)
        y = np.random.randint(5, 10, size=(BATCH_SIZE, query_maxlen)).astype(np.int32)
        expected = model.predict([x, y])
        self.assertTrue(run_onnx_runtime(onnx_model.graph.name, onnx_model, {model.input_names[0]: x, model.input_names[1]: y}, expected, self.model_files))

    @unittest.skipIf(is_tensorflow_older_than('2.0.0'), "Result is slightly different in tf1")
    @unittest.skipIf(get_maximum_opset_supported() < 9,
                     "None seq_length LSTM is not supported before opset 9.")
    def test_imdb_bidirectional_lstm(self):
        # A Bidirectional LSTM on the IMDB sentiment classification task.
        # from https://github.com/keras-team/keras/blob/master/examples/imdb_bidirectional_lstm.py
        max_features = 20000
        maxlen = 100
        batch_size = 32
        model = Sequential()
        model.add(Embedding(max_features, 128, input_length=maxlen))
        model.add(Bidirectional(LSTM(64)))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))
        onnx_model = mock_keras2onnx.convert_keras(model, model.name)
        x = np.random.rand(batch_size, maxlen).astype(np.float32)
        expected = model.predict(x)
        self.assertTrue(run_onnx_runtime(onnx_model.graph.name, onnx_model, x, expected, self.model_files))

    def test_imdb_cnn_lstm(self):
        # A recurrent convolutional network on the IMDB sentiment classification task.
        # from https://github.com/keras-team/keras/blob/master/examples/imdb_cnn_lstm.py
        max_features = 20000
        maxlen = 100
        embedding_size = 128
        kernel_size = 5
        filters = 64
        pool_size = 4
        lstm_output_size = 70
        batch_size = 30

        model = Sequential()
        model.add(Embedding(max_features, embedding_size, input_length=maxlen))
        model.add(Dropout(0.25))
        model.add(Conv1D(filters,
                         kernel_size,
                         padding='valid',
                         activation='relu',
                         strides=1))
        model.add(MaxPooling1D(pool_size=pool_size))
        model.add(LSTM(lstm_output_size))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))
        onnx_model = mock_keras2onnx.convert_keras(model, model.name)
        x = np.random.rand(batch_size, maxlen).astype(np.float32)
        expected = model.predict(x)
        self.assertTrue(run_onnx_runtime(onnx_model.graph.name, onnx_model, x, expected, self.model_files))

    @unittest.skipIf(get_maximum_opset_supported() < 9,
                     "None seq_length LSTM is not supported before opset 9.")
    def test_imdb_lstm(self):
        # An LSTM model on the IMDB sentiment classification task.
        # from https://github.com/keras-team/keras/blob/master/examples/imdb_lstm.py
        max_features = 20000
        maxlen = 80
        batch_size = 32
        model = Sequential()
        model.add(Embedding(max_features, 128))
        model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
        model.add(Dense(1, activation='sigmoid'))
        onnx_model = mock_keras2onnx.convert_keras(model, model.name)
        x = np.random.rand(batch_size, maxlen).astype(np.float32)
        expected = model.predict(x)
        self.assertTrue(run_onnx_runtime(onnx_model.graph.name, onnx_model, x, expected, self.model_files))

    def test_lstm_text_generation(self):
        # Generate text from Nietzsche's writings.
        # from https://github.com/keras-team/keras/blob/master/examples/lstm_text_generation.py
        maxlen = 40
        chars_len = 20
        batch_size = 32
        model = Sequential()
        model.add(LSTM(128, input_shape=(maxlen, chars_len)))
        model.add(Dense(chars_len, activation='softmax'))
        onnx_model = mock_keras2onnx.convert_keras(model, model.name)
        x = np.random.rand(batch_size, maxlen, chars_len).astype(np.float32)
        expected = model.predict(x)
        self.assertTrue(run_onnx_runtime(onnx_model.graph.name, onnx_model, x, expected, self.model_files))

    def test_reuters_mlp(self):
        # An MLP on the Reuters newswire topic classification task.
        # from https://github.com/keras-team/keras/blob/master/examples/reuters_mlp.py
        max_words = 1000
        batch_size = 32
        num_classes = 20
        model = Sequential()
        model.add(Dense(512, input_shape=(max_words,)))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes))
        model.add(Activation('softmax'))
        onnx_model = mock_keras2onnx.convert_keras(model, model.name)
        x = np.random.rand(batch_size, max_words).astype(np.float32)
        expected = model.predict(x)
        self.assertTrue(run_onnx_runtime(onnx_model.graph.name, onnx_model, x, expected, self.model_files))


    if __name__ == "__main__":
        unittest.main()
