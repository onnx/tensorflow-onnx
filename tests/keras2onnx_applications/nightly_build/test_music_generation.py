# SPDX-License-Identifier: Apache-2.0

import os
import sys
import unittest
import mock_keras2onnx
import numpy as np
import tensorflow as tf
from onnxconverter_common.onnx_ex import get_maximum_opset_supported
from mock_keras2onnx.proto import keras
from os.path import dirname, abspath
sys.path.insert(0, os.path.join(dirname(abspath(__file__)), '../../keras2onnx_tests/'))
from test_utils import run_keras_and_ort, test_level_0
K = keras.backend

Activation = keras.layers.Activation
AveragePooling2D = keras.layers.AveragePooling2D
Add = keras.layers.Add
BatchNormalization = keras.layers.BatchNormalization
Concatenate = keras.layers.Concatenate
Conv1D = keras.layers.Conv1D
Dense = keras.layers.Dense
Dropout = keras.layers.Dropout
Embedding = keras.layers.Embedding
Flatten = keras.layers.Flatten
GlobalAveragePooling2D = keras.layers.GlobalAveragePooling2D
Input = keras.layers.Input
Lambda = keras.layers.Lambda
LeakyReLU = keras.layers.LeakyReLU
LSTM = keras.layers.LSTM
MaxPooling2D = keras.layers.MaxPooling2D
multiply = keras.layers.multiply
Permute = keras.layers.Permute
RepeatVector = keras.layers.RepeatVector
Reshape = keras.layers.Reshape
SeparableConv2D = keras.layers.SeparableConv2D
TimeDistributed = keras.layers.TimeDistributed
UpSampling2D = keras.layers.UpSampling2D
ZeroPadding2D = keras.layers.ZeroPadding2D

Sequential = keras.models.Sequential
Model = keras.models.Model


# Define the musical styles
genre = [
    'baroque',
    'classical',
    'romantic'
]

styles = [
    [
        'data/baroque/bach',
        'data/baroque/handel',
        'data/baroque/pachelbel'
    ],
    [
        'data/classical/burgmueller',
        'data/classical/clementi',
        'data/classical/haydn',
        'data/classical/beethoven',
        'data/classical/brahms',
        'data/classical/mozart'
    ],
    [
        'data/romantic/balakirew',
        'data/romantic/borodin',
        'data/romantic/brahms',
        'data/romantic/chopin',
        'data/romantic/debussy',
        'data/romantic/liszt',
        'data/romantic/mendelssohn',
        'data/romantic/moszkowski',
        'data/romantic/mussorgsky',
        'data/romantic/rachmaninov',
        'data/romantic/schubert',
        'data/romantic/schumann',
        'data/romantic/tchaikovsky',
        'data/romantic/tschai'
    ]
]

NUM_STYLES = sum(len(s) for s in styles)

# MIDI Resolution
DEFAULT_RES = 96
MIDI_MAX_NOTES = 128
MAX_VELOCITY = 127

# Number of octaves supported
NUM_OCTAVES = 4
OCTAVE = 12

# Min and max note (in MIDI note number)
MIN_NOTE = 36
MAX_NOTE = MIN_NOTE + NUM_OCTAVES * OCTAVE
NUM_NOTES = MAX_NOTE - MIN_NOTE

# Number of beats in a bar
BEATS_PER_BAR = 4
# Notes per quarter note
NOTES_PER_BEAT = 4
# The quickest note is a half-note
NOTES_PER_BAR = NOTES_PER_BEAT * BEATS_PER_BAR

# Training parameters
BATCH_SIZE = 16
SEQ_LEN = 8 * NOTES_PER_BAR

# Hyper Parameters
OCTAVE_UNITS = 64
STYLE_UNITS = 64
NOTE_UNITS = 3
TIME_AXIS_UNITS = 256
NOTE_AXIS_UNITS = 128

TIME_AXIS_LAYERS = 2
NOTE_AXIS_LAYERS = 2

def one_hot(i, nb_classes):
    arr = np.zeros((nb_classes,))
    arr[i] = 1
    return arr

def pitch_pos_in_f(time_steps):
    """
    Returns a constant containing pitch position of each note
    """
    def f(x):
        note_ranges = tf.range(NUM_NOTES, dtype='float32') / NUM_NOTES
        repeated_ranges = tf.tile(note_ranges, [tf.shape(x)[0] * time_steps])
        return tf.reshape(repeated_ranges, [tf.shape(x)[0], time_steps, NUM_NOTES, 1])
    return f

def pitch_class_in_f(time_steps):
    """
    Returns a constant containing pitch class of each note
    """
    def f(x):
        pitch_class_matrix = np.array([one_hot(n % OCTAVE, OCTAVE) for n in range(NUM_NOTES)])
        pitch_class_matrix = tf.constant(pitch_class_matrix, dtype='float32')
        pitch_class_matrix = tf.reshape(pitch_class_matrix, [1, 1, NUM_NOTES, OCTAVE])
        return tf.tile(pitch_class_matrix, [tf.shape(x)[0], time_steps, 1, 1])
    return f

def pitch_bins_f(time_steps):
    def f(x):
        bins = tf.reduce_sum([x[:, :, i::OCTAVE, 0] for i in range(OCTAVE)], axis=3)
        bins = tf.tile(bins, [NUM_OCTAVES, 1, 1])
        bins = tf.reshape(bins, [tf.shape(x)[0], time_steps, NUM_NOTES, 1])
        return bins
    return f

def time_axis(dropout):
    def f(notes, beat, style):
        time_steps = int(notes.get_shape()[1])

        # TODO: Experiment with when to apply conv
        note_octave = TimeDistributed(Conv1D(OCTAVE_UNITS, 2 * OCTAVE, padding='same'))(notes)
        note_octave = Activation('tanh')(note_octave)
        note_octave = Dropout(dropout)(note_octave)

        # Create features for every single note.
        note_features = Concatenate()([
            Lambda(pitch_pos_in_f(time_steps))(notes),
            Lambda(pitch_class_in_f(time_steps))(notes),
            Lambda(pitch_bins_f(time_steps))(notes),
            note_octave,
            TimeDistributed(RepeatVector(NUM_NOTES))(beat)
        ])

        x = note_features

        # [batch, notes, time, features]
        x = Permute((2, 1, 3))(x)

        # Apply LSTMs
        for l in range(TIME_AXIS_LAYERS):
            # Integrate style
            style_proj = Dense(int(x.get_shape()[3]))(style)
            style_proj = TimeDistributed(RepeatVector(NUM_NOTES))(style_proj)
            style_proj = Activation('tanh')(style_proj)
            style_proj = Dropout(dropout)(style_proj)
            style_proj = Permute((2, 1, 3))(style_proj)
            x = Add()([x, style_proj])

            x = TimeDistributed(LSTM(TIME_AXIS_UNITS, return_sequences=True))(x)
            x = Dropout(dropout)(x)

        # [batch, time, notes, features]
        return Permute((2, 1, 3))(x)
    return f

def note_axis(dropout):
    dense_layer_cache = {}
    lstm_layer_cache = {}
    note_dense = Dense(2, activation='sigmoid', name='note_dense')
    volume_dense = Dense(1, name='volume_dense')

    def f(x, chosen, style):
        time_steps = int(x.get_shape()[1])

        # Shift target one note to the left.
        shift_chosen = Lambda(lambda x: tf.pad(x[:, :, :-1, :], [[0, 0], [0, 0], [1, 0], [0, 0]]))(chosen)

        # [batch, time, notes, 1]
        shift_chosen = Reshape((time_steps, NUM_NOTES, -1))(shift_chosen)
        # [batch, time, notes, features + 1]
        x = Concatenate(axis=3)([x, shift_chosen])

        for l in range(NOTE_AXIS_LAYERS):
            # Integrate style
            '''
            if l not in dense_layer_cache:
                dense_layer_cache[l] = Dense(int(x.get_shape()[3]))
            '''
            dense_layer_cache[0] = Dense(259)
            dense_layer_cache[1] = Dense(128)

            style_proj = dense_layer_cache[l](style)
            style_proj = TimeDistributed(RepeatVector(NUM_NOTES))(style_proj)
            style_proj = Activation('tanh')(style_proj)
            style_proj = Dropout(dropout)(style_proj)
            x = Add()([x, style_proj])

            if l not in lstm_layer_cache:
                lstm_layer_cache[l] = LSTM(NOTE_AXIS_UNITS, return_sequences=True)

            x = TimeDistributed(lstm_layer_cache[l])(x)
            x = Dropout(dropout)(x)

        return Concatenate()([note_dense(x), volume_dense(x)])
    return f

def build_models(time_steps=SEQ_LEN, input_dropout=0.2, dropout=0.5):
    notes_in = Input((time_steps, NUM_NOTES, NOTE_UNITS))
    beat_in = Input((time_steps, NOTES_PER_BAR))
    style_in = Input((time_steps, NUM_STYLES))
    # Target input for conditioning
    chosen_in = Input((time_steps, NUM_NOTES, NOTE_UNITS))

    # Dropout inputs
    notes = Dropout(input_dropout)(notes_in)
    beat = Dropout(input_dropout)(beat_in)
    chosen = Dropout(input_dropout)(chosen_in)

    # Distributed representations
    style_l = Dense(STYLE_UNITS, name='style')
    style = style_l(style_in)

    """ Time axis """
    time_out = time_axis(dropout)(notes, beat, style)

    """ Note Axis & Prediction Layer """
    naxis = note_axis(dropout)
    notes_out = naxis(time_out, chosen, style)

    model = Model([notes_in, chosen_in, beat_in, style_in], [notes_out])

    """ Generation Models """
    time_model = Model([notes_in, beat_in, style_in], [time_out])

    note_features = Input((1, NUM_NOTES, TIME_AXIS_UNITS), name='note_features')
    chosen_gen_in = Input((1, NUM_NOTES, NOTE_UNITS), name='chosen_gen_in')
    style_gen_in = Input((1, NUM_STYLES), name='style_in')

    # Dropout inputs
    chosen_gen = Dropout(input_dropout)(chosen_gen_in)
    style_gen = style_l(style_gen_in)

    note_gen_out = naxis(note_features, chosen_gen, style_gen)

    note_model = Model([note_features, chosen_gen_in, style_gen_in], note_gen_out)

    return model, time_model, note_model


# Model from https://github.com/calclavia/DeepJ
class TestMusicGeneration(unittest.TestCase):

    def setUp(self):
        self.model_files = []

    def tearDown(self):
        for fl in self.model_files:
            os.remove(fl)

    @unittest.skipIf(get_maximum_opset_supported() < 10,
                     "ScatterNd support need opset >= 10.")
    def test_music_generation(self):
        K.clear_session()
        model, time_model, note_model = build_models()

        batch_size = 2
        data_notes = np.random.rand(batch_size, SEQ_LEN, NUM_NOTES, NOTE_UNITS).astype(np.float32)
        data_beat = np.random.rand(batch_size, SEQ_LEN, NOTES_PER_BAR).astype(np.float32)
        data_style = np.random.rand(batch_size, SEQ_LEN, NUM_STYLES).astype(np.float32)
        data_chosen = np.random.rand(batch_size, SEQ_LEN, NUM_NOTES, NOTE_UNITS).astype(np.float32)

        expected = model.predict([data_notes, data_chosen, data_beat, data_style])
        onnx_model = mock_keras2onnx.convert_keras(model, model.name)
        self.assertTrue(
            run_keras_and_ort(onnx_model.graph.name, onnx_model, model,
                              {model.input_names[0]: data_notes,
                               model.input_names[1]: data_chosen,
                               model.input_names[2]: data_beat,
                               model.input_names[3]: data_style}, expected, self.model_files))

        expected = time_model.predict([data_notes, data_beat, data_style])
        onnx_model = mock_keras2onnx.convert_keras(time_model, time_model.name)
        self.assertTrue(
            run_keras_and_ort(onnx_model.graph.name, onnx_model, time_model,
                              {time_model.input_names[0]: data_notes,
                               time_model.input_names[1]: data_beat,
                               time_model.input_names[2]: data_style}, expected, self.model_files))

        data_notes = np.random.rand(batch_size, 1, NUM_NOTES, TIME_AXIS_UNITS).astype(np.float32)
        data_chosen = np.random.rand(batch_size, 1, NUM_NOTES, NOTE_UNITS).astype(np.float32)
        data_style = np.random.rand(batch_size, 1, NUM_STYLES).astype(np.float32)
        expected = note_model.predict([data_notes, data_chosen, data_style])
        onnx_model = mock_keras2onnx.convert_keras(note_model, note_model.name)
        self.assertTrue(
            run_keras_and_ort(onnx_model.graph.name, onnx_model, note_model,
                              {note_model.input_names[0]: data_notes,
                               note_model.input_names[1]: data_chosen,
                               note_model.input_names[2]: data_style}, expected, self.model_files))


if __name__ == "__main__":
    unittest.main()
