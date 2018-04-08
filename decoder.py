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
from tensorflow.python.layers import core as layers_core


def create_attention_mechanism(attention_option, num_units, memory,
                               source_sequence_length):
    """Create attention mechanism based on the attention_option."""
    # Mechanism

    if attention_option == "luong":
        attention_mechanism = tf.contrib.seq2seq.LuongAttention(
            num_units, memory, memory_sequence_length=source_sequence_length)
    elif attention_option == "scaled_luong":
        attention_mechanism = tf.contrib.seq2seq.LuongAttention(
            num_units,
            memory,
            memory_sequence_length=source_sequence_length,
            scale=True)
    elif attention_option == "bahdanau":
        attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
            num_units, memory, memory_sequence_length=source_sequence_length)
    elif attention_option == "normed_bahdanau":
        attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
            num_units,
            memory,
            memory_sequence_length=source_sequence_length,
            normalize=True)
    else:
        raise ValueError("Unknown attention option %s" % attention_option)

    return attention_mechanism


class Decoder:

    def __init__(self, hparams, mode, dtype=tf.float32, scope=None):
        self.hparams = hparams
        self.mode = mode
        self.dtype = dtype
        self.decoder_scope = scope
        # Projection
        # with tf.variable_scope(scope or "build_network"):
        self.output_layer = layers_core.Dense(
            hparams.tgt_vocab_size, use_bias=True, name="output_projection")
        self.cell = self._build_graph(hparams)

    def _build_graph(self, hparams):
        dropout = hparams.dropout if self.mode == tf.contrib.learn.ModeKeys.TRAIN else 0.0  ##??

        with tf.variable_scope("decoder_cell") as scope:
            # Cell Type
            if hparams.unit_type == "lstm":
                utils.print_out("  LSTM, forget_bias=%g" % hparams.forget_bias, new_line=False)
                cell = tf.contrib.rnn.BasicLSTMCell(
                    hparams.num_units,
                    forget_bias=hparams.forget_bias)
            elif hparams.unit_type == "gru":
                utils.print_out("  GRU", new_line=False)
                cell = tf.contrib.rnn.GRUCell(hparams.num_units)
            else:
                raise ValueError("Required decoder cell not supported!")

            # Wrap dropout to encoder cell
            if dropout > 0.0:
                cell = tf.contrib.rnn.DropoutWrapper(
                    cell=cell, input_keep_prob=(1.0 - dropout))

            # Add residual to encoder cell
            if hparams.residual:
                cell = tf.contrib.rnn.ResidualWrapper(cell)

            # Device Wrapper
            # if hparams.encoder_device:
            #     cell = tf.contrib.rnn.DeviceWrapper(cell, hparams.encoder_device)
            # self.decoder_scope = scope
            return cell

    def tile_batch(self, encoder_outputs, encoder_state, source_sequence_length, batch_size):
        encoder_outputs = tf.contrib.seq2seq.tile_batch(encoder_outputs, multiplier=self.hparams.beam_width)
        encoder_state = tf.contrib.seq2seq.tile_batch(encoder_state, multiplier=self.hparams.beam_width)
        source_sequence_length = tf.contrib.seq2seq.tile_batch(source_sequence_length,
                                                               multiplier=self.hparams.beam_width)
        batch_size = batch_size * self.hparams.beam_width
        return encoder_outputs, encoder_state, source_sequence_length, batch_size

    def decode(self,
               encoder_outputs,
               encoder_state,
               source_sequence_length,
               target_inputs,
               target_sequence_length,
               embedding_decoder,
               tgt_sos_id,
               tgt_eos_id):
        if self.hparams.time_major:
            encoder_outputs = tf.transpose(encoder_outputs, [1, 0, 2])
        if self.mode == tf.contrib.learn.ModeKeys.INFER and self.hparams.beam_width > 0:
            encoder_outputs, encoder_state, source_sequence_length, batch_size = self.tile_batch(encoder_outputs,
                                                                                                 encoder_state,
                                                                                                 source_sequence_length,
                                                                                                 self.hparams.batch_size)
        else:
            batch_size = self.hparams.batch_size

        attention_mechanism = create_attention_mechanism(self.hparams.attention,
                                                         self.hparams.num_units,
                                                         encoder_outputs,
                                                         source_sequence_length)
        # print attention_mechanism.scope_name
        # Only generate alignment in greedy INFER mode.
        alignment_history = (self.mode == tf.contrib.learn.ModeKeys.INFER and
                             self.hparams.beam_width == 0)
        self.cell = tf.contrib.seq2seq.AttentionWrapper(self.cell,
                                                        attention_mechanism,
                                                        attention_layer_size=self.hparams.num_units,
                                                        alignment_history=alignment_history,
                                                        name="attention")
        # print self.cell.scope_name
        if self.hparams.pass_hidden_state:
            decoder_initial_state = self.cell.zero_state(batch_size, self.dtype).clone(
                cell_state=encoder_state[0])
        else:
            decoder_initial_state = self.cell.zero_state(batch_size, self.dtype)

        if self.mode == tf.contrib.learn.ModeKeys.TRAIN or self.mode == tf.contrib.learn.ModeKeys.EVAL:
            # Helper
            if self.hparams.time_major:
                target_inputs = tf.transpose(target_inputs, [1, 0, 2])
            helper = tf.contrib.seq2seq.TrainingHelper(
                target_inputs, target_sequence_length,
                time_major=self.hparams.time_major)

            # Decoder

            decoder = tf.contrib.seq2seq.BasicDecoder(
                self.cell,
                helper,
                decoder_initial_state, )

            # Dynamic decoding
            print (self.decoder_scope.name)
            outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(
                decoder,
                output_time_major=self.hparams.time_major,
                swap_memory=True,
                scope=self.decoder_scope)
            with tf.variable_scope(self.decoder_scope):
                logits = self.output_layer(outputs.rnn_output)
            print (self.output_layer.scope_name)
            print (self.cell.scope_name)
            sample_id = outputs.sample_id
        elif self.mode == tf.contrib.learn.ModeKeys.INFER:
            beam_width = self.hparams.beam_width
            length_penalty_weight = self.hparams.length_penalty_weight
            start_tokens = tf.fill([self.hparams.batch_size], tgt_sos_id)
            end_token = tgt_eos_id

            if beam_width > 0:
                my_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                    cell=self.cell,
                    embedding=embedding_decoder,
                    start_tokens=start_tokens,
                    end_token=end_token,
                    initial_state=decoder_initial_state,
                    beam_width=beam_width,
                    output_layer=self.output_layer,
                    length_penalty_weight=length_penalty_weight)
            else:
                # Helper
                helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                    embedding_decoder, start_tokens, end_token)

                # Decoder
                my_decoder = tf.contrib.seq2seq.BasicDecoder(
                    self.cell,
                    helper,
                    decoder_initial_state,
                    output_layer=self.output_layer  # applied per timestep
                )
            decoding_length_factor = 1.5  # 最大解码长度比
            max_encoder_length = tf.reduce_max(source_sequence_length)
            maximum_iterations = tf.to_int32(tf.round(
                tf.to_float(max_encoder_length) * decoding_length_factor))
            # Dynamic decoding
            print (self.decoder_scope.name)
            outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(
                my_decoder,
                maximum_iterations=maximum_iterations,
                output_time_major=self.hparams.time_major,
                swap_memory=True,
                scope=self.decoder_scope)
            print self.cell.scope_name

            print (self.output_layer.scope_name)
            if beam_width > 0:
                logits = tf.no_op()
                sample_id = outputs.predicted_ids
            else:
                logits = outputs.rnn_output
                sample_id = outputs.sample_id
        else:
            raise ValueError("Decoder decode: mode not supported!")
        return logits, sample_id, final_context_state
