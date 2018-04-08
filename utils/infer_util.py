# coding=utf-8
'''
@author:Xin Huang
@contact:xin.huang@nlpr.ia.ac.cn
@file:infer_util.py
@time:2018/4/210:58
@desc:
'''

import tensorflow as tf
import misc_utils as utils
import evaluation_utils
import numpy as np


def get_translation(nmt_outputs, sent_id, tgt_eos, bpe_delimiter):
    """Given batch decoding outputs, select a sentence and turn to text."""
    if tgt_eos: tgt_eos = tgt_eos.encode("utf-8")
    if bpe_delimiter: bpe_delimiter = bpe_delimiter.encode("utf-8")
    # Select a sentence
    output = nmt_outputs[sent_id, :].tolist()

    # If there is an eos symbol in outputs, cut them at that point.
    if tgt_eos and tgt_eos in output:
        output = output[:output.index(tgt_eos)]

    if not bpe_delimiter:
        translation = utils.format_text(output)
    else:  # BPE
        translation = utils.format_bpe_text(output, delimiter=bpe_delimiter)

    return translation


def run_infer(hparams, sess, model, current_epoch, metrics):
    trans_file = hparams.out_dir + "/translation.txt"
    trans_f = open(trans_file, 'w')
    with model.graph.as_default():
        saver = model.saver
        latest_chkpt = tf.train.latest_checkpoint(hparams.out_dir)
        print (latest_chkpt)
        saver.restore(sess, latest_chkpt)
        if current_epoch == 0:
            sess.run(tf.tables_initializer())

    sess.run(model.iterator.initializer)
    num_sentences = 0
    printlogits = True
    while True:
        try:
            logits, final_context_state, sample_id, nmt_outputs, source = model.model.infer(sess)
            # print(np.shape(nmt_outputs))
            # nmt_outputs = np.transpose(nmt_outputs, [2, 0, 1])
            # print (nmt_outputs)
            if printlogits:
                printlogits = False
                # print (logits)
                # print final_context_state
                # print sample_id
            if hparams.time_major:
                nmt_outputs = np.transpose(nmt_outputs)
            if hparams.beam_width > 0:
                # get the top translation.
                nmt_outputs = nmt_outputs[0]

            num_sentences += len(nmt_outputs)
            for sent_id in range(len(nmt_outputs)):
                translation = get_translation(
                    nmt_outputs,
                    sent_id,
                    tgt_eos=hparams.eos,
                    bpe_delimiter=hparams.bpe_delimiter)
                trans_f.write((translation + b"\n"))
        except tf.errors.OutOfRangeError:
            break
    # print (source[0])
    # print (nmt_outputs[0])

    trans_f.close()
    print ("sentence number: " + str(num_sentences))
    # Evaluation
    evaluation_scores = {}
    ref_file = "%s.%s" % (hparams.test_prefix, hparams.tgt)
    sample_print("%s.%s" % (hparams.dev_prefix, hparams.src),
                 trans_file,
                 ref_file)
    if ref_file and tf.gfile.Exists(trans_file):
        for metric in metrics:
            score = evaluation_utils.evaluate(
                ref_file,
                trans_file,
                metric,
                bpe_delimiter=hparams.bpe_delimiter)
            evaluation_scores[metric] = score
            utils.print_out("  %s %s: %.1f" % (metric, "test", score))


def sample_print(source, nmt_output, ref):
    with open(source, 'r') as fp:
        print "source: " + fp.readline().strip()
    with open(nmt_output, 'r') as fp:
        print "nmt   : " + fp.readline().strip()
    with open(ref, 'r') as fp:
        print "ref   : " + fp.readline().strip()