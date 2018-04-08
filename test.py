# coding=utf-8
'''
@author:Xin Huang
@contact:xin.huang@nlpr.ia.ac.cn
@file:test.py
@time:2018/3/2721:01
@desc:
'''
import tensorflow as tf


def model_save_test():

    graph = tf.Graph()
    with graph.as_default():
        with tf.variable_scope("v1"):
            v1 = tf.Variable
    pass

if __name__ == "__main__":
    model_save_test()