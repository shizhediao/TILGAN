import cPickle
import pdb
import os
import glob

# word, vocab = cPickle.load(open('./'+vocab_file))
word, vocab = cPickle.load(
                    open('./vocab_coco_14.pkl', 'rb'))
# input_file = 'save/coco_451.txt'

input_file = './train_coco_14.txt.index'  # syn_val_words
output_file = './train.txt'

with open(output_file, 'w')as fout:
    with open(input_file)as fin:
        for line in fin:
            #line.decode('utf-8')
            line = line.split()
            #line.pop() 
            #line.pop()                 
            line = [int(x) for x in line]
            line = [word[x] for x in line if x != 0]
            # if 'OTHERPAD' not in line:
            line = ' '.join(line) + '\n'
            fout.write(line)#.encode('utf-8'))
