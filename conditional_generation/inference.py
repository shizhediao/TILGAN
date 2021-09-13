from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import sys
import time
import numpy as np
import tensorflow as tf
import os
import re

import data_utils
from data_utils import *
import argparse
from model import TILGAN
import collections
from gensim.models import KeyedVectors
FLAGS = None

# tf.enable_eager_execution()
def add_arguments(parser):
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument("--data_dir", type=str, default="data/", help="Data directory")
    parser.add_argument("--model_dir", type=str, default="inference_ckpt_example/", help="Model directory")
    parser.add_argument("--out_dir", type=str, default="output/", help="Out directory")
    parser.add_argument("--train_dir", type=str, default="tilgan/", help="Training directory")
    parser.add_argument("--gpu_device", type=str, default="0", help="which gpu to use")
    parser.add_argument("--train_data", type=str, default="training",
                        help="Training data path")
    parser.add_argument("--valid_data", type=str, default="dev",
                        help="Valid data path")
    parser.add_argument("--test_data", type=str, default="test",
                        help="Test data path")
    parser.add_argument("--from_vocab", type=str, default="data/vocab_20000",
                        help="from vocab path")
    parser.add_argument("--to_vocab", type=str, default="data/vocab_20000",
                        help="to vocab path")
    parser.add_argument("--output_dir", type=str, default="tfm/")
    parser.add_argument("--max_train_data_size", type=int, default=0, help="Limit on the size of training data (0: no limit)")
    parser.add_argument("--from_vocab_size", type=int, default=20000, help="source vocabulary size")
    parser.add_argument("--to_vocab_size", type=int, default=20000, help="target vocabulary size")
    parser.add_argument("--num_layers", type=int, default=6, help="Number of layers in the model")
    parser.add_argument("--num_units", type=int, default=512, help="Size of each model layer")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of heads in attention")
    parser.add_argument("--emb_dim", type=int, default=300, help="Dimension of word embedding")
    parser.add_argument("--latent_dim", type=int, default=64, help="Dimension of latent variable")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size to use during training")
    parser.add_argument("--max_gradient_norm", type=float, default=3.0, help="Clip gradients to this norm")
    parser.add_argument("--learning_rate_decay_factor", type=float, default=0.5, help="Learning rate decays by this much")
    parser.add_argument("--learning_rate", type=float, default=1, help="Learning rate")
    parser.add_argument("--dropout_rate", type=float, default=0.15, help="Dropout rate")
    parser.add_argument("--epoch_num", type=int, default=100, help="Number of epoch")


def create_hparams(flags):
    return tf.contrib.training.HParams(
        # dir path
        data_dir=flags.data_dir,
        train_dir=flags.train_dir,
        output_dir=flags.output_dir,

        # data params
        batch_size=flags.batch_size,
        from_vocab_size=flags.from_vocab_size,
        to_vocab_size=flags.to_vocab_size,
        GO_ID=data_utils.GO_ID,
        EOS_ID=data_utils.EOS_ID,
        PAD_ID=data_utils.PAD_ID,

        train_data=flags.train_data,
        valid_data=flags.valid_data,
        test_data=flags.test_data,

        from_vocab=flags.from_vocab,
        to_vocab=flags.to_vocab,

        dropout_rate=flags.dropout_rate,
        init_weight=0.1,
        emb_dim=flags.emb_dim,
        latent_dim=flags.latent_dim,
        num_units=flags.num_units,
        num_heads=flags.num_heads,
        num_layers=flags.num_layers,
        learning_rate=flags.learning_rate,
        clip_value=flags.max_gradient_norm,
        decay_factor=flags.learning_rate_decay_factor,
        epoch_num=flags.epoch_num,
    )

def get_config_proto(log_device_placement=False, allow_soft_placement=True):
  config_proto = tf.ConfigProto(
      log_device_placement=log_device_placement,
      allow_soft_placement=allow_soft_placement)
  config_proto.gpu_options.allow_growth = True
  return config_proto


class TrainModel(
    collections.namedtuple("TrainModel",
                           ("graph", "model"))):
  pass

class EvalModel(
    collections.namedtuple("EvalModel",
                           ("graph", "model"))):
  pass

class InferModel(
    collections.namedtuple("InferModel",
                           ("graph", "model"))):
  pass

def create_model(hparams, model, length=22):
    train_graph = tf.Graph()
    with train_graph.as_default():
        train_model = model(hparams, tf.contrib.learn.ModeKeys.TRAIN)

    eval_graph = tf.Graph()
    with eval_graph.as_default():
        eval_model = model(hparams, tf.contrib.learn.ModeKeys.EVAL)

    infer_graph = tf.Graph()
    with infer_graph.as_default():
        infer_model = model(hparams, tf.contrib.learn.ModeKeys.INFER)

    return TrainModel(graph=train_graph, model=train_model), EvalModel(graph=eval_graph, model=eval_model), InferModel(
        graph=infer_graph, model=infer_model)

def read_data(src_path):
    data_set = []
    counter = 0
    max_length1 = 0
    with tf.gfile.GFile(src_path, mode="r") as src_file:
        src = src_file.readline()
        while src:
            if counter % 100000 == 0:
                print("  reading data line %d" % counter)
                sys.stdout.flush()

            sentences = []
            s = []
            for x in src.split(" "):
                id = int(x)
                if id != -1:
                    s.append(id)
                else:
                    if len(s) > max_length1:
                        max_length1 = len(s)
                    sentences.append(s)
                    s = []

            data_set.append(sentences)
            counter += 1
            src = src_file.readline()
    print(counter)
    print(max_length1)
    return data_set


def safe_exp(value):
  """Exponentiation with catching of overflow error."""
  try:
    ans = math.exp(value)
  except OverflowError:
    ans = float("inf")
  return ans

def get_inference_position(path):
    with open(path,'r') as f:
        position_list = []
        for line in f:
            sentences = line.strip().split(".")
            if (sentences[0].strip() == "<pending_infer>"):
                position = 0
            elif (sentences[1].strip() == "<pending_infer>"):
                position = 1
            elif (sentences[2].strip() == "<pending_infer>"):
                position = 2
            elif (sentences[3].strip() == "<pending_infer>"):
                position = 3
            else:
                position = 4
            position_list.append(position)
    return position_list
def train(hparams):
    embeddings = init_embedding(hparams)
    hparams.add_hparam(name="embeddings", value=embeddings)
    print("Vocab load over.")
    train_model, eval_model, infer_model = create_model(hparams, TILGAN)
    config = get_config_proto(
        log_device_placement=False)
    train_sess = tf.Session(config=config, graph=train_model.graph)
    eval_sess = tf.Session(config=config, graph=eval_model.graph)
    infer_sess = tf.Session(config=config, graph=infer_model.graph)
    print("Model create over.")
    #train_data = read_data("data/train.ids")
    valid_data = read_data("data/valid.ids")
    #test_data = read_data("data/test.ids")
    infer_data = read_data("inference_data/infer.ids")
    position_list = get_inference_position("inference_data/infer.txt")
    print("begin")
    print(position_list)
    ckpt = tf.train.get_checkpoint_state(hparams.train_dir)
    ckpt_path = os.path.join(hparams.train_dir, "ckpt")
    with train_model.graph.as_default():
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
            train_model.model.saver.restore(train_sess, ckpt.model_checkpoint_path)
            #eval_model.model.saver.restore(eval_sess, ckpt.model_checkpoint_path)
            infer_model.model.saver.restore(infer_sess, ckpt.model_checkpoint_path)
            global_step = train_model.model.global_step.eval(session=train_sess)
        else:
            raise ValueError("Inference ckpt does not exist")
    to_vocab, rev_to_vocab = data_utils.initialize_vocabulary(hparams.from_vocab)

    step_loss, step_time, total_predict_count, total_loss, total_time, avg_loss, avg_time = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    total_loss_disc, total_loss_gen, total_loss_gan_ae,avg_disc_loss, avg_gen_loss ,avg_gan_ae_loss= 0.0, 0.0, 0.0, 0.0,0.0,0.0
    
    inference_out_f = open("inference/" + "predict2_file" + str(global_step),"w", encoding="utf-8")

    for id in range(0, int(len(infer_data) / hparams.batch_size)):
        given, answer, predict = infer_model.model.infer_step(infer_sess, infer_data, no_random=True,
                                                                id=id* hparams.batch_size,position=position_list[id])
        for i in range(hparams.batch_size):
            sample_output = predict[i]
            if hparams.EOS_ID in sample_output:
                sample_output = sample_output[:sample_output.index(hparams.EOS_ID)]
            pred = []
            for output in sample_output:
                pred.append(tf.compat.as_str(rev_to_vocab[output]))

            sample_output = answer[i]
            if hparams.EOS_ID in sample_output[:]:
                if sample_output[0] == hparams.GO_ID:
                    sample_output = sample_output[1:sample_output.index(hparams.EOS_ID)]
                else:
                    sample_output = sample_output[0:sample_output.index(hparams.EOS_ID)]
            ans = []
            for output in sample_output:
                ans.append(tf.compat.as_str(rev_to_vocab[output]))
            if id < 8:
                #print("answer: ", " ".join(ans))
                print("predict: ", " ".join(pred))

            inference_out_f.write(" ".join(pred) + "\n")
    inference_out_f.close()
    print("infer done.")

def init_embedding(hparams):
    f = open("data/vocab_20000", "r", encoding="utf-8")
    vocab = []
    for line in f:
        vocab.append(line.rstrip("\n"))
    # word_vectors = KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin", binary=True)

    word_vectors = KeyedVectors.load_word2vec_format("data/roc_vector.txt")
    emb = []
    num = 0
    for i in range(0, len(vocab)):
        word = vocab[i]
        if word in word_vectors:
            num += 1
            emb.append(word_vectors[word])
        else:
            emb.append((0.1 * np.random.random([hparams.emb_dim]) - 0.05).astype(np.float32))

    print(" init embedding finished")
    emb = np.array(emb)
    print(num)
    print(emb.shape)
    return emb

def main(_):
    hparams = create_hparams(FLAGS)
    train(hparams)

if __name__ == "__main__":
    my_parser = argparse.ArgumentParser()
    add_arguments(my_parser)
    FLAGS, remaining = my_parser.parse_known_args()
    FLAGS.train_dir = FLAGS.model_dir + FLAGS.train_dir
    FLAGS.output_dir = FLAGS.out_dir + FLAGS.output_dir
    print(FLAGS)
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu_device
    tf.app.run()
