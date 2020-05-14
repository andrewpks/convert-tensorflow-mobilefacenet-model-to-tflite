# -*- coding: utf-8 -*-

import os
import tensorflow as tf
from tensorflow.python.framework import graph_util
from nets.MobileFaceNet import inference

ckpt_files_dir = "./output/ckpt_best/mobilefacenet_best_ckpt/"

def parser_ckpt_path(ckpt_dir):
    dir_list = ckpt_dir.split('/')
    if dir_list[-1] == '':
        dir_list.pop()

    ckpt_dir_pre = "/".join(dir_list[:-1])
    ckpt_dir_pre_name = dir_list[-1]

    ckpt_evl_dir_name = ckpt_dir_pre_name + "_evl"
    ckpt_evl_dir = os.path.join(ckpt_dir_pre, ckpt_evl_dir_name)

    ckpt_state = tf.train.get_checkpoint_state(ckpt_dir)
    ckpt_path = ckpt_state.model_checkpoint_path

    ckpt_folder, ckpt_name = os.path.split(ckpt_path)
    filename, extension = os.path.splitext(ckpt_name)
    ckpt_evl_path = os.path.join(ckpt_evl_dir, ckpt_name)

    return ckpt_path, ckpt_evl_path, filename

def do_ckpt_eval(ckpt_path, ckpt_evl_path):
    with tf.Graph().as_default():
        inputs = tf.placeholder(name='img_inputs', shape=[None, 112, 112, 3], dtype=tf.float32)
        prelogits, net_points = inference(inputs)
        embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            saver = tf.train.Saver(tf.trainable_variables())

            traning_checkpoint = ckpt_path
            eval_checkpoint = ckpt_evl_path
            saver.restore(sess, traning_checkpoint)
            save_path = saver.save(sess, eval_checkpoint)
            print("Model saved in file: %s" % save_path)

def ckpt2pb(ckpt_evl_path, filename):
    ckpt_evl_dir, ckpt_evl_name = os.path.split(ckpt_evl_path)

    # Before exporting our graph, we need to precise what is our output node
    # this variables is plural, because you can have multiple output nodes
    output_node_names = "embeddings"

    # We clear the devices, to allow TensorFlow to control on the loading where it wants operations to be calculated
    clear_devices = True

    # We import the meta graph and retrive a Saver
    saver = tf.train.import_meta_graph(ckpt_evl_path + '.meta', clear_devices=clear_devices)

    # We retrieve the protobuf graph definition
    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()

    #We start a session and restore the graph weights
    with tf.Session() as sess:
        saver.restore(sess, ckpt_evl_path)

        # We use a built-in TF helper to export variables to constant
        output_graph_def = graph_util.convert_variables_to_constants(
            sess,
            input_graph_def,
            output_node_names.split(",") # We split on comma for convenience
        )

        # Finally we serialize and dump the output graph to the filesystem
        output_graph = os.path.join(ckpt_evl_dir, filename + ".pb")
        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))
        print(output_graph)

    return output_graph

def pb2tflite(output_pb_path, filename):
    pb_dir, pb_name = os.path.split(output_pb_path)
    inputs=["img_inputs"]
    classes=["embeddings"]
    converter = tf.contrib.lite.TFLiteConverter.from_frozen_graph(output_pb_path, inputs, classes)
    tflite_model=converter.convert()
    tflite_file = os.path.join(pb_dir, filename + ".tflite")
    open(tflite_file, "wb").write(tflite_model)
    print(tflite_file)

if __name__ == '__main__':
    ckpt_state = tf.train.get_checkpoint_state(ckpt_files_dir)
    print(type(ckpt_state))
    if ckpt_state == None:
        print("Invalid check point files path!")
        exit(0)
    ckpt_path, ckpt_evl_path, filename = parser_ckpt_path(ckpt_files_dir)
    do_ckpt_eval(ckpt_path, ckpt_evl_path)
    output_pb_path = ckpt2pb(ckpt_evl_path, filename)
    pb2tflite(output_pb_path, filename)