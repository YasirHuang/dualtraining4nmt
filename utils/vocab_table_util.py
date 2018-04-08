# coding=utf-8
'''
@author:Xin Huang
@contact:xin.huang@nlpr.ia.ac.cn
@file:vocab_table_util.py
@time:2018/3/288:57
@desc:
'''
from utils import misc_utils as utils
from tensorflow.python.ops import lookup_ops
import codecs
import tensorflow as tf
import os

UNK = "<unk>"
SOS = "<s>"
EOS = "</s>"
UNK_ID = 0


def check_vocab(vocab_file, out_dir, sos=None, eos=None, unk=None):
    """Check if vocab_file doesn't exist, create from corpus_file."""
    utils.print_out(vocab_file)
    if tf.gfile.Exists(vocab_file):
        utils.print_out("# Vocab file %s exists" % vocab_file)
        vocab = []
        with codecs.getreader("utf-8")(tf.gfile.GFile(vocab_file, "rb")) as f:
            vocab_size = 0
            for word in f:
                vocab_size += 1
                vocab.append(word.strip())

        # Verify if the vocab starts with unk, sos, eos
        # If not, prepend those tokens & generate a new vocab file
        if not unk: unk = UNK
        if not sos: sos = SOS
        if not eos: eos = EOS
        assert len(vocab) >= 3
        if vocab[0] != unk or vocab[1] != sos or vocab[2] != eos:
            utils.print_out("The first 3 vocab words [%s, %s, %s]"
                            " are not [%s, %s, %s]" %
                            (vocab[0], vocab[1], vocab[2], unk, sos, eos))
            vocab = [unk, sos, eos] + vocab
            vocab_size += 3
            new_vocab_file = os.path.join(out_dir, os.path.basename(vocab_file))
            with codecs.getwriter("utf-8")(tf.gfile.GFile(new_vocab_file, "wb")) as f:
                for word in vocab:
                    f.write("%s\n" % word)
            vocab_file = new_vocab_file
    else:
        raise ValueError("vocab_file does not exist.")

    vocab_size = len(vocab)
    return vocab_size, vocab_file


def get_vocab_table(src_vocab_file, tgt_vocab_file):
    src_vocab_table = lookup_ops.index_table_from_file(src_vocab_file, default_value=UNK_ID)
    tgt_vocab_table = lookup_ops.index_table_from_file(tgt_vocab_file, default_value=UNK_ID)
    return src_vocab_table, tgt_vocab_table
