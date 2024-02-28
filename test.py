from extract_training_data import FeatureExtractor, State

WORD_VOCAB_FILE = 'data/words.vocab'
POS_VOCAB_FILE = 'data/pos.vocab'
word_vocab_f = open(WORD_VOCAB_FILE,'r')
pos_vocab_f = open(POS_VOCAB_FILE,'r')

extractor = FeatureExtractor(word_vocab_f, pos_vocab_f)
state = State(range(1,10))
me = extractor.get_input_representation(['I', 'like', 'cats'], ['PRP', 'VBP', 'NNS'], state)