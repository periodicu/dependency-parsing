from conll_reader import DependencyStructure, DependencyEdge, conll_reader
from collections import defaultdict
import copy
import sys

import numpy as np
import keras

from extract_training_data import FeatureExtractor, State

class Parser(object): 

    def __init__(self, extractor, modelfile):
        self.model = keras.models.load_model(modelfile)
        self.extractor = extractor
        
        # The following dictionary from indices to output actions will be useful
        self.output_labels = dict([(index, action) for (action, index) in extractor.output_labels.items()])

    def parse_sentence(self, words, pos):
        state = State(range(1,len(words)))
        state.stack.append(0)    

        while state.buffer:
            # ed #124
            features = self.extractor.get_input_representation(
                words, pos, state
                ).reshape((1, -1))

            posacts = self.model.predict(features)

            # (action, probability)
            posacts_tup = []
            for i in range(91):
                action = self.output_labels[i]
                probability = posacts[0][i]
                posacts_tup.append((action, probability))

            #x: x[1] ?? maybe
            sorted_posacts = sorted(posacts_tup, key=lambda x: x[1], reverse=True)

            for ((action, label), _) in sorted_posacts:
                # print(action)
                if action == 'shift':
                    if len(state.buffer) > 1:
                        state.shift()
                        break
                    elif len(state.buffer) == 1 and len(state.stack) == 0:
                        state.shift()
                        break
                elif action == 'left_arc':
                    # can't left arc on root
                    if state.stack and state.stack[-1] != 0:
                        state.left_arc(label)
                        break
                elif action == 'right_arc':
                    if state.stack:
                        state.right_arc(label)
                        break

        result = DependencyStructure()
        for p,c,r in state.deps: 
            result.add_deprel(DependencyEdge(c,words[c],pos[c],p, r))
        return result 
        

if __name__ == "__main__":

    WORD_VOCAB_FILE = 'data/words.vocab'
    POS_VOCAB_FILE = 'data/pos.vocab'

    try:
        word_vocab_f = open(WORD_VOCAB_FILE,'r')
        pos_vocab_f = open(POS_VOCAB_FILE,'r') 
    except FileNotFoundError:
        print("Could not find vocabulary files {} and {}".format(WORD_VOCAB_FILE, POS_VOCAB_FILE))
        sys.exit(1) 

    extractor = FeatureExtractor(word_vocab_f, pos_vocab_f)
    parser = Parser(extractor, sys.argv[1])

    with open(sys.argv[2],'r') as in_file: 
        for dtree in conll_reader(in_file):
            words = dtree.words()
            pos = dtree.pos()
            deps = parser.parse_sentence(words, pos)
            print("bajs")
            print(deps.print_conll())
            print()
        
