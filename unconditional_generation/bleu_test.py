# from utils.metrics.Bleu import Bleu
from pycocoevalcap.bleu.bleu import Bleu
import pdb
import numpy as np

import nltk
import re
import pdb
import numpy as np
import time

from pycocoevalcap.bleu.bleu import Bleu
def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", "s", string)
    string = re.sub(r"\'ve", " have", string)
    string = re.sub(r"n\'t", " not", string)
    string = re.sub(r"\'re", " are", string)
    string = re.sub(r"\'ll", " will", string)
    string = re.sub(r"\(", "", string)
    string = re.sub(r"\)", "", string)
    string = re.sub(r"\'", "", string)
    string = re.sub(r",", "", string)
    string = re.sub(r"!", "", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip() if TREC else string.strip().lower()


def score(ref, hypo):
    """
    ref, dictionary of reference sentences (id, sentence)
    hypo, dictionary of hypothesis sentences (id, sentence)
    score, dictionary of scores
    """
    scorers = [
        (Bleu(5), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4", "Bleu_5"])
    ]
    final_scores = {}
    for scorer, method in scorers:
        score, scores = scorer.compute_score(ref, hypo)
        if type(score) == list:
            for m, s in zip(method, score):
                final_scores[m] = s
        else:
            final_scores[method] = score
    return final_scores

def bleu_test(real_text, test_text):
    real_text_sents = []
    test_text_sents = []
    with open(real_text, 'r')as fin:
        for line in fin:
            real_text_sents.append(line)
    with open(test_text, 'r')as fin:
        for line in fin:
            test_text_sents.append(line)

    ans = np.zeros(5)
    for i in range(len(test_text_sents)):
        hop = {0: [test_text_sents[i]]}
        ref = {0: real_text_sents}

        ans[4] += score(ref, hop)['Bleu_5']
        ans[3] += score(ref, hop)['Bleu_4']
        ans[2] += score(ref, hop)['Bleu_3']
        ans[1] += score(ref, hop)['Bleu_2']
        ans[0] += score(ref, hop)['Bleu_1']

    ans /= len(test_text_sents)
    print("bleu_test: ", ans)
    return ans

if __name__ == "__main__":
    # real_text = './data/snli_lm/test.txt'  # real
    # test_text = './results/klgan_examples_bsz64_epoch15_v2/013_examplar_gen'  # syn_val_words.txt
    real_text = "./data/NewsData/test.txt"
    test_text = "./results/025_examplar_gen"
    start = time.time()
    ans = bleu_test(real_text, test_text)
    print("elapsed time = ", time.time() - start)



