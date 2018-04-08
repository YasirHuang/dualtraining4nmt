# coding=utf-8
'''
@author:Xin Huang
@contact:xin.huang@nlpr.ia.ac.cn
@file:decoder.py
@time:2018/3/2614:55
@desc:
'''
import tensorflow as tf
import numpy as np
import config
import argparse
import os
import sys
import time
import model_creator as mc
import model_helper
from utils import vocab_table_util
from utils import misc_utils as utils
from utils import infer_util
from utils import eval_util


def train(hparams):
    train_model = mc.create_train_model(hparams)
    eval_model = mc.create_eval_model(hparams)
    infer_model = mc.create_infer_model(hparams)
    # TensorFlow model
    config_proto = utils.get_config_proto(
        log_device_placement=hparams.log_device_placement, allow_soft_placement=True)
    train_sess = tf.Session(graph=train_model.graph, config=config_proto)
    eval_sess = tf.Session(graph=eval_model.graph, config=config_proto)
    infer_sess = tf.Session(graph=infer_model.graph, config=config_proto)

    with train_model.graph.as_default():
        train_model, global_step = model_helper.create_or_load_model(train_model, hparams.out_dir, train_sess, "train")
    train_sess.run(train_model.iterator.initializer)

    step_in_epoch = 0
    current_epoch = 0
    chkpt_predicted_count, chkpt_train_loss = 0.0, 0.0

    start_time = time.time()
    while current_epoch < hparams.epoch:
        try:
            _, \
            train_loss, \
            step_predicted_count, \
            source, \
            target_input, \
            target_output, \
            logits, \
            final_context_state, \
            sample_id, \
            global_step = train_model.model.train(train_sess)
            step_in_epoch += 1
        except tf.errors.OutOfRangeError:
            step_in_epoch = 0
            print ("epoch %s finished" % str(current_epoch))

            infer_util.run_infer(hparams, infer_sess, infer_model, current_epoch, ["bleu", "rouge", "accuracy"])

            current_epoch = current_epoch + 1
            train_sess.run(train_model.iterator.initializer)
            continue

        # train_loss = step_results[0]
        # step_predicted_count = step_results[1]

        chkpt_predicted_count += step_predicted_count
        chkpt_train_loss += (train_loss * hparams.batch_size)

        if step_in_epoch % hparams.steps_per_eval == 0 and step_in_epoch > 0:
            # if hparams.time_major:
            #     source = np.transpose(source)
            #     target_input = np.transpose(target_input)
            #     target_output = np.transpose(target_output)
            print ("global step: ", str(global_step))
            # print (source[0])
            # print (target_input[0])
            # print (target_output[0])
            # print (np.shape(logits))
            # print (final_context_state)
            # print (sample_id)

            # user_input = input("please input")
            # print (user_input)
            train_ppl = utils.safe_exp(chkpt_train_loss / chkpt_predicted_count)
            print ("epoch %d: step_in_epoch %d, train ppl %.2f, avg loss %.2f, lr %.7f, time %.2fs"
                   % (current_epoch,
                      step_in_epoch,
                      train_ppl,
                      chkpt_train_loss/(hparams.steps_per_eval * hparams.batch_size),
                      train_model.model.learning_rate.eval(session=train_sess),
                      time.time() - start_time))
            start_time = time.time()
            chkpt_predicted_count, chkpt_train_loss = 0.0, 0.0

            train_model.saver.save(train_sess, os.path.join(hparams.out_dir, "translate.ckpt"), global_step=current_epoch)

            eval_ppl, eval_loss = eval_util.run_eval(hparams, eval_sess, eval_model, current_epoch, step_in_epoch, )
            # infer_util.run_infer(hparams, infer_sess, infer_model, current_epoch, ["bleu", "rouge", "accuracy"])



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
    train(hparams)
    pass


if __name__ == "__main__":
    nmt_parser = argparse.ArgumentParser()
    config.add_arguments(nmt_parser)
    FLAGS, unparsed = nmt_parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
