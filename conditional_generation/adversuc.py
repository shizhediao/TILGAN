

from ernie import SentenceClassifier, Models
import pandas as pd
import sys

import tensorflow as tf
import sys
from tensorflow.python.platform import gfile

def read_data(src_path):
    data_set = []
    counter = 0
    max_length1 = 0
    with tf.io.gfile.GFile(src_path, mode="r") as src_file:
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

def initialize_vocabulary(vocabulary_path):
  if gfile.Exists(vocabulary_path):
    rev_vocab = []
    with gfile.GFile(vocabulary_path, mode="r") as f:
      rev_vocab.extend(f.readlines())
    rev_vocab = [line.strip() for line in rev_vocab]
    vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
    return vocab, rev_vocab
  else:
    raise ValueError("Vocabulary file %s not found.", vocabulary_path)

def get_pred(path):
    prediction=[]
    file = open(path,'r') #'./output/predict2_file105000'
    line = file.read().splitlines()
    while line:
        prediction.append(line)
        line = file.read().splitlines()
    file.close()
    return prediction  #Using prediction[0][i] to see each sentence

def convert_pred_to_story(data,pred_sentence,converter):
    word = []
    output = []
    pred_story = []
    for i in range (0,len(pred_sentence[0])):
        for j in range (0,len(data[i])):
            for k in range (0,len(data[i][j])):
                word.append(tf.compat.as_str(converter[data[i][j][k]]))
            new_sentence = " ".join(word)
            if j == i % 5:
                output.append(pred_sentence[0][i])
            else:
                output.append(new_sentence)

            word.clear()
        new_stroy = " ".join(output)
        pred_story.append(new_stroy)
        output.clear()


    return pred_story



def main():
    #Preprocessing
    path_pred_file = sys.argv[1]
    print(path_pred_file)
    valid_data = read_data("data/valid.ids")
    to_vocab, rev_to_vocab = initialize_vocabulary("data/vocab_20000")
    pred_sentence = get_pred(path_pred_file)
    pred_story = convert_pred_to_story(valid_data,pred_sentence,rev_to_vocab)
    print(pred_story[0])
    
    #Model evaluation
    classifier = SentenceClassifier(model_path='./model_adver_suc')
    probabilities = classifier.predict(pred_story)
    total_count = 0
    pos_count = 0
    for prob in probabilities:
        if prob[1] > 0.50:
            pos_count += 1
        total_count +=1
        if (total_count % 100 == 0):
            print(total_count)
            print("AdverSuc : %f" %(pos_count / total_count))
    print("positive count : %d" %pos_count)
    print("AdverSuc : %f" %(pos_count / total_count))
    
if __name__ == "__main__":
    main()