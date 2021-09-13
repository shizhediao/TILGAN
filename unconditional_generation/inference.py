import os
import time
import math
import numpy as np
import random
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

from utils import to_gpu, Corpus, batchify, train_ngram_lm, get_ppl, create_exp_dir
from models import Seq2Seq, MLP_D, MLP_D_local, MLP_G
from bleu_self import *
from bleu_test import *
import datetime
now_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

import argparse
parser = argparse.ArgumentParser(description='TILGAN for unconditional generation')
parser.add_argument('--data_path', type=str, required=True,
                    help='location of the data corpus')
parser.add_argument('--save', type=str, default='result',
                    help='trained model ckpt directory')
#Data Processing Arguments
parser.add_argument('--maxlen', type=int, default=15,
                    help='maximum length')
parser.add_argument('--vocab_size', type=int, default=0,
                    help='cut vocabulary down to this size '
                         '(most frequently seen words in train)')
parser.add_argument('--lowercase', dest='lowercase', action='store_true',
                    help='lowercase all text')
parser.add_argument('--no-lowercase', dest='lowercase', action='store_true',
                    help='not lowercase all text')
parser.set_defaults(lowercase=True)
# Model Arguments
parser.add_argument('--emsize', type=int, default=512,
                    help='size of word embeddings')
parser.add_argument('--nhidden', type=int, default=512,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--noise_r', type=float, default=0.05,
                    help='stdev of noise for autoencoder (regularizer)')
parser.add_argument('--nheads', type=int, default=4,
                    help='number of heads')
parser.add_argument('--nff', type=int, default=1024,
                    help='feedforward network dimension in Transformer')
parser.add_argument('--aehidden', type=int, default=56,
                    help='the squeezed hidden dimension')
parser.add_argument('--noise_anneal', type=float, default=0.9995,
                    help='anneal noise_r exponentially by this'
                         'every 100 iterations')
parser.add_argument('--hidden_init', action='store_true',
                    help="initialize decoder hidden state with encoder's")
parser.add_argument('--arch_g', type=str, default='300-300',
                    help='generator architecture (MLP)')
parser.add_argument('--arch_d', type=str, default='300-300',
                    help='critic/discriminator architecture (MLP)')
parser.add_argument('--arch_d_local', type=str, default='300-300',
                    help='local critic/discriminator architecture (MLP)')
parser.add_argument('--z_size', type=int, default=100,
                    help='dimension of random noise z to feed into generator')
parser.add_argument('--dropout', type=float, default=0.3,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--noise_seq_length', type=int, default=15,
                    help='the seq_length of fake noise ')
parser.add_argument('--gan_type', type=str, default='kl', choices=['kl', 'all', 'wgan'],
                    help='generator architecture (MLP)')

# Training Arguments
parser.add_argument('--epochs', type=int, default=100,
                    help='maximum number of epochs')
parser.add_argument('--min_epochs', type=int, default=12,
                    help="minimum number of epochs to train for")
parser.add_argument('--no_earlystopping', action='store_true',
                    help="won't use KenLM for early stopping")
parser.add_argument('--patience', type=int, default=2,
                    help="number of language model evaluations without ppl "
                         "improvement to wait before early stopping")
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='batch size')
parser.add_argument('--eval_batch_size', type=int, default=32,
                    help='eval batch size')
parser.add_argument('--niters_ae', type=int, default=1,
                    help='number of autoencoder iterations in training')
parser.add_argument('--niters_gan_d', type=int, default=2,
                    help='number of discriminator iterations in training')
parser.add_argument('--niters_gan_dec', type=int, default=1,
                    help='number of enhance decoder')
parser.add_argument('--niters_gan_g', type=int, default=1,
                    help='number of generator iterations in training')
parser.add_argument('--niters_gan_ae', type=int, default=2,
                    help='number of gan-into-ae iterations in training')
parser.add_argument('--niters_gan_schedule', type=str, default='',
                    help='epoch counts to increase number of GAN training '
                         ' iterations (increment by 1 each time)')
parser.add_argument('--lr_ae', type=float, default=0.001,
                    help='autoencoder learning rate')
parser.add_argument('--lr_gan_e', type=float, default=1e-04,
                    help='gan encoder learning rate')
parser.add_argument('--lr_gan_g', type=float, default=5e-05,
                    help='generator learning rate')
parser.add_argument('--lr_gan_d', type=float, default=1e-04,
                    help='critic/discriminator learning rate')
parser.add_argument('--beta1', type=float, default=0.5,
                    help='beta1 for adam. default=0.5')
parser.add_argument('--clip', type=float, default=1,
                    help='gradient clipping, max norm')
parser.add_argument('--gan_clamp', type=float, default=0.01,
                    help='WGAN clamp')
parser.add_argument('--gan_gp_lambda', type=float, default=1,
                    help='WGAN GP penalty lambda')
parser.add_argument('--gan_lambda', type=float, default=0.1,
                    help='coefficient of divergence (minimized with GAN)')
parser.add_argument('--add_noise', action='store_true',
                    help='whether to add_noise, default is False')
parser.add_argument('--gan_d_local', action='store_true',
                    help='whether to turn on gan_d_local, default is False')
parser.add_argument('--gan_d_local_windowsize', type=int, default=3,
                    help='gan_d_local_windowsize')
parser.add_argument('--gan_g_activation', action='store_true',
                    help='whether to turn on activation of gan_g, default is False')
parser.add_argument('--enhance_dec', action='store_true',
                    help='whether to enhance decoder')

# Evaluation Arguments
parser.add_argument('--sample', action='store_true',
                    help='sample when decoding for generation')
parser.add_argument('--N', type=int, default=5,
                    help='N-gram order for training n-gram language model')
parser.add_argument('--log_interval', type=int, default=200,
                    help='interval to log autoencoder training results')

# Other
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
args = parser.parse_args()
print(vars(args))

corpus = Corpus(args.data_path,
                maxlen=args.maxlen,
                vocab_size=args.vocab_size,
                lowercase=args.lowercase)

# save arguments
ntokens = len(corpus.dictionary.word2idx)
print("Vocabulary Size: {}".format(ntokens))
args.ntokens = ntokens

autoencoder = Seq2Seq(add_noise=args.add_noise,
                      emsize=args.emsize,
                      nhidden=args.nhidden,
                      ntokens=args.ntokens,
                      nlayers=args.nlayers,
                      nheads=args.nheads,
                      nff=args.nff,
                      aehidden=args.aehidden,
                      noise_r=args.noise_r,
                      hidden_init=args.hidden_init,
                      dropout=args.dropout,
                      gpu=True)
nlatent = args.aehidden * (args.maxlen+1)
gan_gen = MLP_G(ninput=args.z_size, noutput=nlatent, layers=args.arch_g, gan_g_activation=args.gan_g_activation)
gan_disc = MLP_D(ninput=nlatent, noutput=1, layers=args.arch_d)
gan_disc_local = MLP_D_local(ninput=args.gan_d_local_windowsize * args.aehidden, noutput=1, layers=args.arch_d_local)

def gen_fixed_noise(noise, to_save):
    gan_gen.eval()
    autoencoder.eval()

    fake_hidden = gan_gen(noise)
    max_indices = autoencoder.generate(fake_hidden, args.maxlen, sample=args.sample)

    with open(to_save, "w") as f:
        max_indices = max_indices.data.cpu().numpy()
        for idx in max_indices:
            # generated sentence
            words = [corpus.dictionary.idx2word[x] for x in idx]
            # truncate sentences to first occurrence of <eos>
            truncated_sent = []
            for w in words:
                if w != '<eos>':
                    truncated_sent.append(w)
                else:
                    break
            chars = " ".join(truncated_sent)
            f.write(chars + '\n')

def load_models():
    model_args = json.load(open(os.path.join(args.save, 'options.json'), 'r'))
    word2idx = json.load(open(os.path.join(args.save, 'vocab.json'), 'r'))
    idx2word = {v: k for k, v in word2idx.items()}

    print('Loading models from {}'.format(args.save))
    loaded = torch.load(os.path.join(args.save, "model.pt"))
    autoencoder.load_state_dict(loaded.get('ae'))
    gan_gen.load_state_dict(loaded.get('gan_g'))
    gan_disc.load_state_dict(loaded.get('gan_d'))
    gan_disc_local.load_state_dict(loaded.get('gan_d_local'))
    return model_args, idx2word, autoencoder, gan_gen, gan_disc,gan_disc_local

model_args, idx2word, autoencoder, gan_gen, gan_disc,gan_disc_local = load_models()
autoencoder = autoencoder.to(device)
gan_gen = gan_gen.to(device)
gan_disc = gan_disc.to(device)
gan_disc_local = gan_disc_local.to(device)
print("Load Ckpt Done")

fixed_noise = Variable(torch.ones(args.eval_batch_size, args.z_size).normal_(0, 1).to(device))
gen_text_savepath = os.path.join(args.save, "inference_result")
gen_fixed_noise(fixed_noise, gen_text_savepath)
print("Infer Done")
