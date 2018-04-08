# coding=utf-8
'''
@author:Xin Huang
@contact:xin.huang@nlpr.ia.ac.cn
@file:decoder.py
@time:2018/3/2614:55
@desc:
'''

import tensorflow as tf
from utils import misc_utils as utils


class Encoder:

    def __init__(self, hparams, mode, dtype=tf.float32, scope=None):
        self.hparams = hparams
        self.mode = mode
        self.dtype = dtype
        cell_fw, cell_bw = self._build_graph(hparams)
        self.cell_fw = cell_fw
        self.cell_bw = cell_bw

    def _build_graph(self, hparams):
        dropout = hparams.dropout if self.mode == tf.contrib.learn.ModeKeys.TRAIN else 0.0  ##??

        with tf.variable_scope("encoder_cell") as scope:
            # Cell Type
            if hparams.unit_type == "lstm":
                utils.print_out("  LSTM, forget_bias=%g" % hparams.forget_bias, new_line=False)
                cell_fw = tf.contrib.rnn.BasicLSTMCell(
                    hparams.num_units,
                    forget_bias=hparams.forget_bias)
                cell_bw = tf.contrib.rnn.BasicLSTMCell(
                    hparams.num_units,
                    forget_bias=hparams.forget_bias)
            elif hparams.unit_type == "gru":
                utils.print_out("  GRU", new_line=False)
                cell_fw = tf.contrib.rnn.GRUCell(hparams.num_units)
                cell_bw = tf.contrib.rnn.GRUCell(hparams.num_units)
            else:
                raise ValueError("Required encoder cell not supported!")

            # Wrap dropout to encoder cell
            if dropout > 0.0:
                cell_fw = tf.contrib.rnn.DropoutWrapper(
                    cell=cell_fw, input_keep_prob=(1.0 - dropout))
                cell_bw = tf.contrib.rnn.DropoutWrapper(
                    cell=cell_bw, input_keep_prob=(1.0 - dropout))

            # Add residual to encoder cell
            if hparams.residual:
                cell_fw = tf.contrib.rnn.ResidualWrapper(cell_fw)
                cell_bw = tf.contrib.rnn.ResidualWrapper(cell_bw)

            # # Device Wrapper
            # if hparams.encoder_device:
            #     cell_fw = tf.contrib.rnn.DeviceWrapper(cell_fw, hparams.encoder_device)
            #     cell_bw = tf.contrib.rnn.DeviceWrapper(cell_bw, hparams.encoder_device)

            return cell_fw, cell_bw

    def encode(self, inputs, sequence_length):
        if self.hparams.time_major:
            inputs = tf.transpose(inputs, [1, 0, 2])
        bi_outputs, bi_state = tf.nn.bidirectional_dynamic_rnn(
            self.cell_fw,
            self.cell_bw,
            inputs,
            dtype=self.dtype,
            sequence_length=sequence_length,
            time_major=self.hparams.time_major)
        # alternatively concat forward and backward states
        return tf.concat(bi_outputs, -1), bi_state
