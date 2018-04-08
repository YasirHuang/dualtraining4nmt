# coding=utf-8
'''
@author:Xin Huang
@contact:xin.huang@nlpr.ia.ac.cn
@file:model_creator.py
@time:2018/3/2815:26
@desc:
'''
import tensorflow as tf
import model as mdl
import encoder as en
import decoder as de
from utils import iterator_utils
from utils import vocab_table_util
from tensorflow.python.ops import lookup_ops
import collections


# Train model
class TrainModel(
    collections.namedtuple("TrainModel", ("graph", "model", "encoder", "decoder", "iterator", "saver"))):
    pass


def create_train_model(hparams):
    # get src/tgt vocabulary table

    graph = tf.Graph()
    with graph.as_default() as graph:
        src_vocab_table, tgt_vocab_table = vocab_table_util.get_vocab_table(hparams.src_vocab_file,
                                                                            hparams.tgt_vocab_file)
        reversed_tgt_vocab_table = lookup_ops.index_to_string_table_from_file(
            hparams.tgt_vocab_file, default_value=vocab_table_util.UNK)
        reversed_src_vocab_table = lookup_ops.index_to_string_table_from_file(
            hparams.src_vocab_file, default_value=vocab_table_util.UNK)
        with tf.variable_scope("NMTModel", initializer=tf.truncated_normal_initializer(stddev=0.01)) as nmtmodel_scope:
            with tf.variable_scope("train_iterator"):
                src_dataset_file = "%s.%s" % (hparams.train_prefix, hparams.src)
                tgt_dataset_file = "%s.%s" % (hparams.train_prefix, hparams.tgt)
                iterator = iterator_utils.get_nmt_iterator(src_dataset_file,
                                                           tgt_dataset_file,
                                                           src_vocab_table,
                                                           tgt_vocab_table,
                                                           hparams.batch_size,
                                                           hparams.eos,
                                                           hparams.sos,
                                                           hparams.source_reverse,
                                                           hparams.random_seed)
            with tf.variable_scope("shared_encoder") as encoder_scope:
                encoder = en.Encoder(hparams, tf.contrib.learn.ModeKeys.TRAIN, dtype=tf.float32, scope=encoder_scope)
            with tf.variable_scope("shared_decoder") as decoder_scope:
                decoder = de.Decoder(hparams, tf.contrib.learn.ModeKeys.TRAIN, dtype=tf.float32, scope=decoder_scope)
            nmt_model = mdl.NMTModel(hparams,
                                     src_vocab_table,
                                     tgt_vocab_table,
                                     encoder,
                                     decoder,
                                     iterator,
                                     tf.contrib.learn.ModeKeys.TRAIN,
                                     reversed_tgt_vocab_table,
                                     reversed_src_vocab_table)
        saver = tf.train.Saver(tf.global_variables())
    return TrainModel(graph=graph,
                      model=nmt_model,
                      encoder=encoder,
                      decoder=decoder,
                      iterator=iterator,
                      saver=saver)


# Evaluation model
class EvalModel(
    collections.namedtuple("EvalModel", ("graph", "model", "encoder", "decoder", "iterator", "saver"))):
    pass


def create_eval_model(hparams):
    # get src/tgt vocabulary table

    graph = tf.Graph()
    with graph.as_default() as graph:
        src_vocab_table, tgt_vocab_table = vocab_table_util.get_vocab_table(hparams.src_vocab_file,
                                                                            hparams.tgt_vocab_file)
        reversed_tgt_vocab_table = lookup_ops.index_to_string_table_from_file(
            hparams.tgt_vocab_file, default_value=vocab_table_util.UNK)
        reversed_src_vocab_table = lookup_ops.index_to_string_table_from_file(
            hparams.src_vocab_file, default_value=vocab_table_util.UNK)
        with tf.variable_scope("NMTModel") as nmtmodel_scope:
            with tf.variable_scope("eval_iterator") as train_iterator_scope:
                src_dataset_file = "%s.%s" % (hparams.dev_prefix, hparams.src)
                tgt_dataset_file = "%s.%s" % (hparams.dev_prefix, hparams.tgt)
                iterator = iterator_utils.get_nmt_iterator(src_dataset_file,
                                                           tgt_dataset_file,
                                                           src_vocab_table,
                                                           tgt_vocab_table,
                                                           hparams.batch_size,
                                                           hparams.eos,
                                                           hparams.sos,
                                                           hparams.source_reverse,
                                                           hparams.random_seed)
            with tf.variable_scope("shared_encoder") as encoder_scope:
                encoder = en.Encoder(hparams, tf.contrib.learn.ModeKeys.EVAL, dtype=tf.float32, scope=encoder_scope)
            with tf.variable_scope("shared_decoder") as decoder_scope:
                decoder = de.Decoder(hparams, tf.contrib.learn.ModeKeys.EVAL, dtype=tf.float32, scope=decoder_scope)
            nmt_model = mdl.NMTModel(hparams,
                                     src_vocab_table,
                                     tgt_vocab_table,
                                     encoder,
                                     decoder,
                                     iterator,
                                     tf.contrib.learn.ModeKeys.EVAL,
                                     reversed_tgt_vocab_table,
                                     reversed_src_vocab_table)
        saver = tf.train.Saver(tf.global_variables())
    return EvalModel(graph=graph,
                     model=nmt_model,
                     encoder=encoder,
                     decoder=decoder,
                     iterator=iterator,
                     saver=saver)


class InferModel(
    collections.namedtuple("InferModel", ("graph", "model", "encoder", "decoder", "iterator", "saver"))):
    pass


def create_infer_model(hparams):
    graph = tf.Graph()
    with graph.as_default() as graph:
        src_vocab_table, tgt_vocab_table = vocab_table_util.get_vocab_table(hparams.src_vocab_file,
                                                                            hparams.tgt_vocab_file)
        reversed_tgt_vocab_table = lookup_ops.index_to_string_table_from_file(
            hparams.tgt_vocab_file, default_value=vocab_table_util.UNK)
        reversed_src_vocab_table = lookup_ops.index_to_string_table_from_file(
            hparams.src_vocab_file, default_value=vocab_table_util.UNK)

        with tf.variable_scope("NMTModel") as nmtmodel_scope:
            with tf.variable_scope("infer_iterator"):
                src_dataset_file = "%s.%s" % (hparams.dev_prefix, hparams.src)
                iterator = iterator_utils.get_nmt_infer_iterator(src_dataset_file,
                                                                 src_vocab_table,
                                                                 hparams.batch_size,
                                                                 hparams.source_reverse,
                                                                 hparams.eos)
            with tf.variable_scope("shared_encoder") as encoder_scope:
                encoder = en.Encoder(hparams, tf.contrib.learn.ModeKeys.INFER, dtype=tf.float32, scope=encoder_scope)
            with tf.variable_scope("shared_decoder") as decoder_scope:
                decoder = de.Decoder(hparams, tf.contrib.learn.ModeKeys.INFER, dtype=tf.float32, scope=decoder_scope)
            nmt_model = mdl.NMTModel(hparams,
                                     src_vocab_table,
                                     tgt_vocab_table,
                                     encoder,
                                     decoder,
                                     iterator,
                                     tf.contrib.learn.ModeKeys.INFER,
                                     reversed_tgt_vocab_table=reversed_tgt_vocab_table,
                                     reversed_src_vocab_table=reversed_src_vocab_table)
        saver = tf.train.Saver(tf.global_variables())
    return InferModel(graph=graph,
                      model=nmt_model,
                      encoder=encoder,
                      decoder=decoder,
                      iterator=iterator,
                      saver=saver)
