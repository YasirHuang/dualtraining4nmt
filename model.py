# coding=utf-8
'''
@author:Xin Huang
@contact:xin.huang@nlpr.ia.ac.cn
@file:model.py.py
@time:2018/3/2621:15
@desc:
'''

import tensorflow as tf
import model_helper


class NMTModel:

    def __init__(self,
                 hparams,
                 src_vocab_table,
                 tgt_vocab_table,
                 encoder,
                 decoder,
                 iterator,
                 mode,
                 reversed_tgt_vocab_table=None,
                 reversed_src_vocab_table=None):
        self.encoder = encoder
        self.decoder = decoder
        self.iterator = iterator
        self.hparams = hparams
        self.src_vocab_table = src_vocab_table
        self.tgt_vocab_table = tgt_vocab_table
        self.mode = mode
        self._initial_embedding()

        if reversed_src_vocab_table is not None and iterator.source is not None:
            self.source = reversed_src_vocab_table.lookup(tf.to_int64(iterator.source))
        if reversed_tgt_vocab_table is not None and self.mode != tf.contrib.learn.ModeKeys.INFER:
            self.target_input = reversed_tgt_vocab_table.lookup(tf.to_int64(iterator.target_input))
            self.target_output = reversed_tgt_vocab_table.lookup(tf.to_int64(iterator.target_output))
        logits, sample_id, final_context_state, loss = self._build_graph(hparams,
                                                                         src_vocab_table,
                                                                         tgt_vocab_table,
                                                                         encoder,
                                                                         decoder,
                                                                         iterator)
        if mode == tf.contrib.learn.ModeKeys.TRAIN:
            self.train_logits = logits
            self.train_sample_id = sample_id
            self.train_final_context_state = final_context_state
            self.train_loss = loss
            self.predicted_count = tf.reduce_sum(iterator.target_sequence_length)
        elif mode == tf.contrib.learn.ModeKeys.EVAL:
            self.eval_loss = loss
            self.predicted_count = tf.reduce_sum(iterator.target_sequence_length)
        else:
            self.infer_logits = logits
            self.infer_sample_id = sample_id
            self.infer_final_context_state = final_context_state
            self.sample_words = reversed_tgt_vocab_table.lookup(tf.to_int64(self.infer_sample_id))

        self.global_step = tf.Variable(0, trainable=False)
        # params = tf.global_variables()
        # use adam optimizer only
        if mode == tf.contrib.learn.ModeKeys.TRAIN:
            if hparams.optimizer == "sgd":
                self.learning_rate = tf.cond(
                    self.global_step < hparams.start_decay_step,
                    lambda: tf.constant(hparams.learning_rate),
                    lambda: tf.train.exponential_decay(
                        hparams.learning_rate,
                        tf.subtract(self.global_step , hparams.start_decay_step),
                        hparams.decay_steps,
                        hparams.decay_factor,
                        staircase=True),
                    name="learning_rate")
                opt = tf.train.GradientDescentOptimizer(self.learning_rate)
                tf.summary.scalar("lr", self.learning_rate)
            elif hparams.optimizer == "adam":
                assert float(
                    hparams.learning_rate
                ) <= 0.001, "! High Adam learning rate %g" % hparams.learning_rate
                self.learning_rate = tf.cond(
                    self.global_step < hparams.start_decay_step,
                    lambda: tf.constant(hparams.learning_rate),
                    lambda: tf.train.exponential_decay(
                        hparams.learning_rate,
                        tf.subtract(self.global_step, hparams.start_decay_step),
                        hparams.decay_steps,
                        hparams.decay_factor,
                        staircase=True),
                    name="learning_rate")
                # self.learning_rate = tf.constant(hparams.learning_rate)
                opt = tf.train.AdamOptimizer(self.learning_rate)
            else:
                raise ValueError("No such optimizer supported.")
            self.update = opt.minimize(loss, global_step=self.global_step)

            # gradients = tf.gradients(
            #     self.train_loss,
            #     params,
            #     colocate_gradients_with_ops=hparams.colocate_gradients_with_ops)
            #
            # clipped_gradients, gradient_norm_summary = model_helper.gradient_clip(
            #     gradients, max_gradient_norm=hparams.max_gradient_norm)
            #
            # self.update = opt.apply_gradients(
            #     zip(clipped_gradients, params))

    def _initial_embedding(self):
        with tf.variable_scope("embeddings", dtype=tf.float32):
            with tf.variable_scope("encoder"):
                self.embedding_source = tf.get_variable(
                    "embedding_source", [self.hparams.src_vocab_size,
                                         self.hparams.num_embedding],
                    tf.float32)

            with tf.variable_scope("decoder"):
                self.embedding_target = tf.get_variable(
                    "embedding_target", [self.hparams.tgt_vocab_size,
                                         self.hparams.num_embedding],
                    tf.float32)

    def get_max_time(self, tensor):
        time_axis = 0 if self.hparams.time_major else 1
        return tensor.shape[time_axis].value or tf.shape(tensor)[time_axis]

    def _compute_loss(self, logits):
        """Compute optimization loss."""
        target_output = self.iterator.target_output
        if self.hparams.time_major:
            target_output = tf.transpose(target_output)
        max_time = self.get_max_time(target_output)
        crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=target_output, logits=logits)
        target_weights = tf.sequence_mask(
            self.iterator.target_sequence_length, max_time, dtype=logits.dtype)
        if self.hparams.time_major:
            target_weights = tf.transpose(target_weights)

        loss = tf.reduce_sum(
            crossent * target_weights) / tf.to_float(self.hparams.batch_size)
        return loss

    def _build_graph(self,
                     hparams,
                     src_vocab_table,
                     tgt_vocab_table,
                     encoder,
                     decoder,
                     iterator):
        source_input = tf.nn.embedding_lookup(self.embedding_source, iterator.source)
        if self.mode != tf.contrib.learn.ModeKeys.INFER:
            target_input = tf.nn.embedding_lookup(self.embedding_target, iterator.target_input)
            target_sequence_length = iterator.target_sequence_length
        else:
            target_input = None
            target_sequence_length = None
        source_sequence_length = iterator.source_sequence_length
        tgt_sos_id = tf.cast(tgt_vocab_table.lookup(tf.constant(hparams.sos)), tf.int32)
        tgt_eos_id = tf.cast(tgt_vocab_table.lookup(tf.constant(hparams.eos)), tf.int32)
        encoder_outputs, encoder_state = encoder.encode(source_input, source_sequence_length)
        logits, sample_id, final_context_state = decoder.decode(
            encoder_outputs,
            encoder_state,
            source_sequence_length,
            target_input,
            target_sequence_length,
            self.embedding_target,
            tgt_sos_id,
            tgt_eos_id
        )
        if self.mode == tf.contrib.learn.ModeKeys.TRAIN or self.mode == tf.contrib.learn.ModeKeys.EVAL:
            loss = self._compute_loss(logits)
        else:
            loss = None
        return logits, sample_id, final_context_state, loss

    def train(self, sess):
        assert self.mode == tf.contrib.learn.ModeKeys.TRAIN
        return sess.run([self.update,
                         self.train_loss,
                         self.predicted_count,
                         self.source,
                         self.target_input,
                         self.target_output,
                         self.train_logits,
                         self.train_final_context_state,
                         self.train_sample_id,
                         self.global_step])

    def eval(self, sess):
        assert self.mode == tf.contrib.learn.ModeKeys.EVAL
        return sess.run([self.eval_loss,
                         self.predicted_count,
                         self.source,
                         self.target_input,
                         self.target_output])

    def infer(self, sess):
        assert self.mode == tf.contrib.learn.ModeKeys.INFER
        return sess.run([self.infer_logits,
                         self.infer_final_context_state,
                         self.infer_sample_id,
                         self.sample_words,
                         self.source])
