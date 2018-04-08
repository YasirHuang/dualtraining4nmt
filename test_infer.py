# coding=utf-8
'''
@author:Xin Huang
@contact:xin.huang@nlpr.ia.ac.cn
@file:test_infer.py
@time:2018/4/310:35
@desc:
'''

import tensorflow as tf
import config
import argparse
import sys
import model_creator as mc
from utils import vocab_table_util
from utils import misc_utils as utils
from utils import infer_util

def infer(hparams):
    infer_model = mc.create_infer_model(hparams)
    # TensorFlow model
    config_proto = utils.get_config_proto(
        log_device_placement=hparams.log_device_placement, allow_soft_placement=True)
    infer_sess = tf.Session(graph=infer_model.graph, config=config_proto)

    infer_util.run_infer(hparams, infer_sess, infer_model, 0, ["bleu", "rouge", "accuracy"])


FLAGS = None
def main(arg):
    hparams = config.create_hparams(FLAGS)
    # Source vocab
    src_vocab_file = "%s.%s" % (hparams.vocab_prefix, hparams.src)
    tgt_vocab_file = "%s.%s" % (hparams.vocab_prefix, hparams.tgt)
    print (src_vocab_file, tgt_vocab_file)
    src_vocab_size, src_vocab_file = vocab_table_util.check_vocab(
        src_vocab_file,
        hparams.out_dir,
        sos=hparams.sos,
        eos=hparams.eos,
        unk=vocab_table_util.UNK)
    # Target vocab
    tgt_vocab_size, tgt_vocab_file = vocab_table_util.check_vocab(
        tgt_vocab_file,
        hparams.out_dir,
        sos=hparams.sos,
        eos=hparams.eos,
        unk=vocab_table_util.UNK)
    hparams.add_hparam("src_vocab_size", src_vocab_size)
    hparams.add_hparam("tgt_vocab_size", tgt_vocab_size)
    hparams.add_hparam("src_vocab_file", src_vocab_file)
    hparams.add_hparam("tgt_vocab_file", tgt_vocab_file)
    infer(hparams)
    pass


if __name__ == "__main__":
    nmt_parser = argparse.ArgumentParser()
    config.add_arguments(nmt_parser)
    FLAGS, unparsed = nmt_parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)