# coding=utf-8
'''
@author:Xin Huang
@contact:xin.huang@nlpr.ia.ac.cn
@file:eval_util.py
@time:2018/4/211:04
@desc:
'''
import tensorflow as tf
import misc_utils as utils

def run_eval(hparams, sess, eval_model, current_epoch, step_in_epoch):
    with eval_model.graph.as_default():
        saver = eval_model.saver
        latest_chkpt = tf.train.latest_checkpoint(hparams.out_dir)
        saver.restore(sess, latest_chkpt)
        if current_epoch == 0 and step_in_epoch == 100:
            sess.run(tf.tables_initializer())

    sess.run(eval_model.iterator.initializer)
    total_loss, total_predicted_count = 0.0, 0.0
    while True:
        try:
            loss, predicted_count, source, target_input, target_output = eval_model.model.eval(sess)
        except tf.errors.OutOfRangeError:
            print("eval finished.")
            break
        total_loss += loss * hparams.batch_size
        total_predicted_count += predicted_count
    ppl = utils.safe_exp(total_loss / total_predicted_count)
    print ("epoch %d: step_in_epoch %d, eval  ppl %.2f" % (current_epoch, step_in_epoch, ppl))
    return ppl, total_loss