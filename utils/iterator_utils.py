# coding=utf-8
'''
@author:Xin Huang
@contact:xin.huang@nlpr.ia.ac.cn
@file:decoder.py
@time:2018/3/2614:55
@desc:
'''
import tensorflow as tf
import collections


class NMTBatchedInput(collections.namedtuple("NMTBatchedInput",
                                             ("initializer",
                                              "source",
                                              "target_input",
                                              "target_output",
                                              "source_sequence_length",
                                              "target_sequence_length"))):
    pass


def get_nmt_infer_iterator(src_dataset_file,
                       src_vocab_table,
                       batch_size,
                       source_reverse,
                       eos,
                       src_max_len=None):
    src_dataset = tf.contrib.data.TextLineDataset(src_dataset_file)
    src_eos_id = tf.cast(src_vocab_table.lookup(tf.constant(eos)), tf.int32)
    src_dataset = src_dataset.map(lambda src: tf.string_split([src]).values)

    if src_max_len:
        src_dataset = src_dataset.map(lambda src: src[:src_max_len])
    # Convert the word strings to ids
    src_dataset = src_dataset.map(
        lambda src: tf.cast(src_vocab_table.lookup(src), tf.int32))
    if source_reverse:
        src_dataset = src_dataset.map(lambda src: tf.reverse(src, axis=[0]))
    # Add in the word counts.
    src_dataset = src_dataset.map(lambda src: (src, tf.size(src)))

    src_dataset = src_dataset.padded_batch(
            batch_size,
            # The entry is the source line rows;
            # this has unknown-length vectors.  The last entry is
            # the source row size; this is a scalar.
            padded_shapes=(tf.TensorShape([None]),  # src
                           tf.TensorShape([])),  # src_len
            # Pad the source sequences with eos tokens.
            # (Though notice we don't generally need to do this since
            # later on we will be masking out calculations past the true sequence.
            padding_values=(src_eos_id,  # src
                            0))  # src_len -- unused

    batched_iter = src_dataset.make_initializable_iterator()
    (src_ids, src_seq_len) = batched_iter.get_next()
    return NMTBatchedInput(
        initializer=batched_iter.initializer,
        source=src_ids,
        target_input=None,
        target_output=None,
        source_sequence_length=src_seq_len,
        target_sequence_length=None)


def get_nmt_iterator(src_dataset_file,
                     tgt_dataset_file,
                     src_vocab_table,
                     tgt_vocab_table,
                     batch_size,
                     eos,
                     sos,
                     source_reverse,
                     random_seed,
                     src_max_len=None,
                     tgt_max_len=None,
                     num_threads=4,
                     output_buffer_size=None,
                     skip_count=None
                     ):
    if not output_buffer_size: output_buffer_size = batch_size * 1000
    src_eos_id = tf.cast(
        src_vocab_table.lookup(tf.constant(eos)),
        tf.int32)
    tgt_sos_id = tf.cast(
        tgt_vocab_table.lookup(tf.constant(sos)),
        tf.int32)
    tgt_eos_id = tf.cast(
        tgt_vocab_table.lookup(tf.constant(eos)),
        tf.int32)
    src_dataset = tf.contrib.data.TextLineDataset(src_dataset_file)
    tgt_dataset = tf.contrib.data.TextLineDataset(tgt_dataset_file)
    src_tgt_dataset = tf.contrib.data.Dataset.zip((src_dataset, tgt_dataset))

    if skip_count is not None:
        src_tgt_dataset = src_tgt_dataset.skip(skip_count)

    src_tgt_dataset = src_tgt_dataset.shuffle(
        output_buffer_size, random_seed)

    src_tgt_dataset = src_tgt_dataset.map(
        lambda src, tgt: (
            tf.string_split([src]).values, tf.string_split([tgt]).values),  # 这一步相当于分词
        num_threads=num_threads,  # 并行处理的个数
        output_buffer_size=output_buffer_size)

    # Filter zero length input sequences.
    src_tgt_dataset = src_tgt_dataset.filter(
        lambda src, tgt: tf.logical_and(tf.size(src) > 0, tf.size(tgt) > 0))

    # 句长过滤
    if src_max_len:
        src_tgt_dataset = src_tgt_dataset.map(
            lambda src, tgt: (src[:src_max_len], tgt),
            num_threads=num_threads,
            output_buffer_size=output_buffer_size)
    if tgt_max_len:
        src_tgt_dataset = src_tgt_dataset.map(
            lambda src, tgt: (src, tgt[:tgt_max_len]),
            num_threads=num_threads,
            output_buffer_size=output_buffer_size)
    if source_reverse:
        src_tgt_dataset = src_tgt_dataset.map(
            lambda src, tgt: (tf.reverse(src, axis=[0]), tgt),
            num_threads=num_threads,
            output_buffer_size=output_buffer_size)
    # Convert the word strings to ids.  Word strings that are not in the
    # vocab get the lookup table's default_value integer.
    src_tgt_dataset = src_tgt_dataset.map(  # 将词都替换成对应的id
        lambda src, tgt: (tf.cast(src_vocab_table.lookup(src), tf.int32),
                          tf.cast(tgt_vocab_table.lookup(tgt), tf.int32)),
        num_threads=num_threads, output_buffer_size=output_buffer_size)
    # Create a tgt_input prefixed with <sos> and a tgt_output suffixed with <eos>.
    src_tgt_dataset = src_tgt_dataset.map(
        lambda src, tgt: (src,
                          tf.concat(([tgt_sos_id], tgt), 0),
                          tf.concat((tgt, [tgt_eos_id]), 0)),
        num_threads=num_threads, output_buffer_size=output_buffer_size)
    # Add in sequence lengths.
    src_tgt_dataset = src_tgt_dataset.map(
        lambda src, tgt_in, tgt_out: (
            src, tgt_in, tgt_out, tf.size(src), tf.size(tgt_in)),
        num_threads=num_threads,
        output_buffer_size=output_buffer_size)

    src_tgt_dataset = src_tgt_dataset.padded_batch(
        batch_size,
        # The first three entries are the source and target line rows;
        # these have unknown-length vectors.  The last two entries are
        # the source and target row sizes; these are scalars.
        padded_shapes=(tf.TensorShape([None]),  # src
                       tf.TensorShape([None]),  # tgt_input
                       tf.TensorShape([None]),  # tgt_output
                       tf.TensorShape([]),  # src_len
                       tf.TensorShape([])),  # tgt_len
        # Pad the source and target sequences with eos tokens.
        # (Though notice we don't generally need to do this since
        # later on we will be masking out calculations past the true sequence.
        padding_values=(src_eos_id,  # src
                        tgt_eos_id,  # tgt_input
                        tgt_eos_id,  # tgt_output
                        0,  # src_len -- unused
                        0))  # tgt_len -- unused

    src_tgt_iter = src_tgt_dataset.make_initializable_iterator()  # 以上都是构造图，然后在这步初始化
    (src_ids, tgt_input_ids, tgt_output_ids, src_seq_len, tgt_seq_len) = (
        src_tgt_iter.get_next())  # 这一步是取出一组数据，但是不是立即执行的，只有在session中run以下才能执行一次
    return NMTBatchedInput(
        initializer=src_tgt_iter.initializer,
        source=src_ids,
        target_input=tgt_input_ids,
        target_output=tgt_output_ids,
        source_sequence_length=src_seq_len,
        target_sequence_length=tgt_seq_len)
