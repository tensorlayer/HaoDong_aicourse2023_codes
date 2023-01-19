# Deprecated, because it use Tensorflow


# Char-level text generation with LSTM
import os
os.environ['TL_BACKEND'] = 'torch'

from tensorlayerx.nn.layers import Embedding, LSTM, Linear
from tensorlayerx.utils import iterate
from tensorlayerx.text import nlp #This requires tensorflow
import tensorlayerx as tlx
import nltk
import numpy as np
import re
import time


init_scale = 0.1
learning_rate = 1e-3
sequence_length = 20
hidden_size = 200
max_epoch = 100
batch_size = 16

top_k_list = [1, 3, 5, 10]
print_length = 30

tlx.set_seed(99999)  # set random set

_UNK = "_UNK"


def basic_clean_str(string):
    """Tokenization/string cleaning for a datasets."""
    string = re.sub(r"\n", " ", string)  # '\n'      --> ' '
    string = re.sub(r"\'s", " \'s", string)  # it's      --> it 's
    string = re.sub(r"\’s", " \'s", string)
    string = re.sub(r"\'ve", " have", string)  # they've   --> they have
    string = re.sub(r"\’ve", " have", string)
    string = re.sub(r"\'t", " not", string)  # can't     --> can not
    string = re.sub(r"\’t", " not", string)
    string = re.sub(r"\'re", " are", string)  # they're   --> they are
    string = re.sub(r"\’re", " are", string)
    string = re.sub(r"\'d", "", string)  # I'd (I had, I would) --> I
    string = re.sub(r"\’d", "", string)
    string = re.sub(r"\'ll", " will", string)  # I'll      --> I will
    string = re.sub(r"\’ll", " will", string)
    string = re.sub(r"\“", "  ", string)  # “a”       --> “ a ”
    string = re.sub(r"\”", "  ", string)
    string = re.sub(r"\"", "  ", string)  # "a"       --> " a "
    string = re.sub(r"\'", "  ", string)  # they'     --> they '
    string = re.sub(r"\’", "  ", string)  # they’     --> they ’
    string = re.sub(r"\.", " . ", string)  # they.     --> they .
    string = re.sub(r"\,", " , ", string)  # they,     --> they ,
    string = re.sub(r"\!", " ! ", string)
    string = re.sub(r"\-", "  ", string)  # "low-cost"--> lost cost
    string = re.sub(r"\(", "  ", string)  # (they)    --> ( they)
    string = re.sub(r"\)", "  ", string)  # ( they)   --> ( they )
    string = re.sub(r"\]", "  ", string)  # they]     --> they ]
    string = re.sub(r"\[", "  ", string)  # they[     --> they [
    string = re.sub(r"\?", "  ", string)  # they?     --> they ?
    string = re.sub(r"\>", "  ", string)  # they>     --> they >
    string = re.sub(r"\<", "  ", string)  # they<     --> they <
    string = re.sub(r"\=", "  ", string)  # easier=   --> easier =
    string = re.sub(r"\;", "  ", string)  # easier;   --> easier ;
    string = re.sub(r"\;", "  ", string)
    string = re.sub(r"\:", "  ", string)  # easier:   --> easier :
    string = re.sub(r"\"", "  ", string)  # easier"   --> easier "
    string = re.sub(r"\$", "  ", string)  # $380      --> $ 380
    string = re.sub(r"\_", "  ", string)  # _100     --> _ 100
    # Akara is    handsome --> Akara is handsome
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()  # lowercase


def customized_clean_str(string):
    """Tokenization/string cleaning for a datasets."""
    string = re.sub(r"\n", " ", string)  # '\n'      --> ' '
    string = re.sub(r"\'s", " \'s", string)  # it's      --> it 's
    string = re.sub(r"\’s", " \'s", string)
    string = re.sub(r"\'ve", " have", string)  # they've   --> they have
    string = re.sub(r"\’ve", " have", string)
    string = re.sub(r"\'t", " not", string)  # can't     --> can not
    string = re.sub(r"\’t", " not", string)
    string = re.sub(r"\'re", " are", string)  # they're   --> they are
    string = re.sub(r"\’re", " are", string)
    string = re.sub(r"\'d", "", string)  # I'd (I had, I would) --> I
    string = re.sub(r"\’d", "", string)
    string = re.sub(r"\'ll", " will", string)  # I'll      --> I will
    string = re.sub(r"\’ll", " will", string)
    string = re.sub(r"\“", " “ ", string)  # “a”       --> “ a ”
    string = re.sub(r"\”", " ” ", string)
    string = re.sub(r"\"", " “ ", string)  # "a"       --> " a "
    string = re.sub(r"\'", " ' ", string)  # they'     --> they '
    string = re.sub(r"\’", " ' ", string)  # they’     --> they '
    string = re.sub(r"\.", " . ", string)  # they.     --> they .
    string = re.sub(r"\,", " , ", string)  # they,     --> they ,
    string = re.sub(r"\-", " ", string)  # "low-cost"--> lost cost
    string = re.sub(r"\(", " ( ", string)  # (they)    --> ( they)
    string = re.sub(r"\)", " ) ", string)  # ( they)   --> ( they )
    string = re.sub(r"\!", " ! ", string)  # they!     --> they !
    string = re.sub(r"\]", " ] ", string)  # they]     --> they ]
    string = re.sub(r"\[", " [ ", string)  # they[     --> they [
    string = re.sub(r"\?", " ? ", string)  # they?     --> they ?
    string = re.sub(r"\>", " > ", string)  # they>     --> they >
    string = re.sub(r"\<", " < ", string)  # they<     --> they <
    string = re.sub(r"\=", " = ", string)  # easier=   --> easier =
    string = re.sub(r"\;", " ; ", string)  # easier;   --> easier ;
    string = re.sub(r"\;", " ; ", string)
    string = re.sub(r"\:", " : ", string)  # easier:   --> easier :
    string = re.sub(r"\"", " \" ", string)  # easier"   --> easier "
    string = re.sub(r"\$", " $ ", string)  # $380      --> $ 380
    string = re.sub(r"\_", " _ ", string)  # _100     --> _ 100
    # Akara is    handsome --> Akara is handsome
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()  # lowercase


def customized_read_words(input_fpath):  # , dictionary):
    with open(input_fpath, "r", encoding="utf8") as f:
        words = f.read()
    # Clean the data
    words = customized_clean_str(words)
    # Split each word
    return words.split()


# Load data
data_path = "fra.txt"

words = customized_read_words(data_path)

vocab = nlp.create_vocab(
    [words], word_counts_output_file='vocab.txt', min_word_count=1)
vocab = nlp.Vocabulary('vocab.txt', unk_word="<UNK>")
vocab_size = vocab.unk_id + 1
train_data = [vocab.word_to_id(word) for word in words]

# # Set the seed to generate sentence.
# seed = "it is a"
# # seed = basic_clean_str(seed).split()
# seed = nltk.tokenize.word_tokenize(seed)
# print('seed : %s' % seed)


class LSTMModel(tlx.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super().__init__()
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.lstm = LSTM(embedding_dim, hidden_size)
        self.dense = Linear(out_features=vocab_size,
                            in_features=hidden_size, act='softmax')

    def forward(self, x, prev_h, prev_c):
        x = self.embedding(x)
        print(x.shape, h.shape)
        x, (h, c) = self.lstm(x, [prev_h, prev_c])
        x = self.dense(x)
        x = tlx.argmax(x, axis=-1)
        return x, h, c


h_init = tlx.convert_to_tensor(np.zeros((1, 20, 256), dtype=np.float32))
c_init = tlx.convert_to_tensor(np.zeros((1, 20, 256), dtype=np.float32))

net = LSTMModel(vocab_size, 256, 256)

train_weights = net.trainable_weights
optimizer = tlx.optimizers.Adam(lr=learning_rate)

print('Training ...')
h_state, c_state = h_init, c_init

for i in range(max_epoch):
    print("Epoch: %d/%d" % (i + 1, max_epoch))
    epoch_size = ((len(train_data) // batch_size) - 1) // sequence_length

    start_time = time.time()
    costs = 0.0
    iters = 0

    net.train()
    # reset all states at the begining of every epoch
    for step, (x, y) in enumerate(iterate.ptb_iterator(train_data, batch_size, sequence_length)):
        # compute outputs
        logits, h_state, c_state = net(tlx.convert_to_tensor(x), h_state, c_state)
        # compute loss and update model
        cost = tlx.losses.binary_cross_entropy(
            logits, tlx.convert_to_tensor(y))

        grad = optimizer.gradient(cost, train_weights)
        optimizer.apply_gradients(zip(grad, train_weights))

        costs += cost
        iters += 1

        if step % (epoch_size // 10) == 1:
            print(
                "%.3f perplexity: %.3f speed: %.0f wps" % (
                    step * 1.0 / epoch_size, np.exp(costs / iters),
                    iters * batch_size * sequence_length *
                    batch_size / (time.time() - start_time)
                )
            )
    train_perplexity = np.exp(costs / iters)
    # print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
    print("Epoch: %d/%d Train Perplexity: %.3f" %
          (i + 1, max_epoch, train_perplexity))
