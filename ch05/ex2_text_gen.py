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

import torch
# device = torch.device('mlu:0' if torch.mlu.is_available() else 'cpu')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# 设置参数
learning_rate = 1e-3 # 学习率
sequence_length = 20 # 序列长度
hidden_size = 512 # 隐藏层大小
embedding_dim = 512 # 词向量维度
max_epoch = 200 # 最大迭代次数
batch_size = 16 # 批大小

top_k_list = [1, 3, 5, 10] # 候选词top k
print_length = 30 # 生成文本长度

tlx.set_seed(99999)  # set random seed

_UNK = "_UNK"

# 文本预处理
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
data_path = "ch05/trump_text.txt" # 川普演讲数据集

words = customized_read_words(data_path)


# 使用nlp工具包中的Vocabulary类来创建词汇表
vocab = nlp.create_vocab(
    [words], word_counts_output_file='ch05/vocab.txt', min_word_count=1)
vocab = nlp.Vocabulary('ch05/vocab.txt', unk_word="<UNK>") # UNK表示未知词
vocab_size = vocab.unk_id + 1
train_data = [vocab.word_to_id(word) for word in words]


# 搭建模型
class LSTMModel(tlx.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super().__init__()
        self.embedding = Embedding(vocab_size, embedding_dim) # 词嵌入层，将词id转换为向量
        self.lstm = LSTM(embedding_dim, hidden_size) # LSTM层
        self.dense = Linear(out_features=vocab_size,
                            in_features=hidden_size, act=None) # 全连接层，将LSTM的输出转换为词表大小的向量

    def forward(self, x, prev_h, prev_c):
        '''
        prev_h: 上一次LSTM的隐藏状态
        prev_c: 上一次LSTM的记忆状态
        '''
        x = self.embedding(x)
        x, (h, c) = self.lstm(x, [prev_h, prev_c]) # LSTM层的输出x的形状为(batch_size, sequence_length, hidden_size)
        x = self.dense(x)
        x = tlx.transpose(x, (0,2,1)) # 将x的形状转换为(batch_size, vocab_size, sequence_length)
        return x, h, c

net = LSTMModel(vocab_size, embedding_dim, hidden_size)

train_weights = net.trainable_weights
optimizer = tlx.optimizers.Adam(lr=learning_rate) # Adam优化器

net = net.to(device)

print('Training ...')

for i in range(max_epoch):
    print("Epoch: %d/%d" % (i + 1, max_epoch))
    epoch_size = ((len(train_data) // batch_size) - 1) // sequence_length # 一个epoch的迭代次数

    start_time = time.time()
    costs = 0.0
    iters = 0

    h_init = tlx.convert_to_tensor(np.zeros((1, 20, hidden_size), dtype=np.float32)).to(device) # 初始化LSTM的隐藏状态
    c_init = tlx.convert_to_tensor(np.zeros((1, 20, hidden_size), dtype=np.float32)).to(device) # 初始化LSTM的记忆状态

    h_state, c_state = h_init, c_init

    # reset all states at the begining of every epoch
    for step, (x, y) in enumerate(iterate.ptb_iterator(train_data, batch_size, sequence_length)): # 循环读取数据
        net.set_train()

        # compute outputs
        logits, h_state, c_state = net(tlx.convert_to_tensor(x).to(device), h_state, c_state) # 前向传播，h_state和c_state会被更新
        # compute loss and update model
        cost = tlx.losses.softmax_cross_entropy_with_logits(
            logits, tlx.convert_to_tensor(y,dtype="int64").to(device)) # 计算损失函数

        grad = optimizer.gradient(cost, train_weights) # 计算梯度
        optimizer.apply_gradients(zip(grad, train_weights))
        
        h_state = tlx.convert_to_tensor( h_state.cpu().detach().numpy()).to(device) # 将h_state和c_state转换为numpy数组，再转换为tensor,避免计算图重复
        c_state = tlx.convert_to_tensor( c_state.cpu().detach().numpy()).to(device)

        costs += cost.cpu().detach().numpy()
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


    net.set_eval() # 设置为测试模式
    # testing: sample from top k words
    # 每个epoch结束后，测试模型的文本生成效果
    for top_k in top_k_list:
        # 设置LSTM的初始状态
        h_init = tlx.convert_to_tensor(np.zeros((1, 1, hidden_size), dtype=np.float32)).to(device)
        c_init = tlx.convert_to_tensor(np.zeros((1, 1, hidden_size), dtype=np.float32)).to(device)

        h_state, c_state = h_init, c_init

        # 测试，根据it is a作为开头生成文本
        outs_id = [vocab.word_to_id(w) for w in ["it", "is", "a"]]
        # 将it is a序列作为开头，作为输入，初始化LSTM的状态
        for ids in outs_id[:-1]:
            a_id = tlx.convert_to_tensor(np.asarray(ids).reshape(1, 1)).to(device)
            _, h_state, c_state = net(a_id, h_state, c_state) # 前向传播，更新LSTM的状态

        # 从it is a的最后一个单词a开始生成文本
        a_id = outs_id[-1]
        for _ in range(print_length):
            a_id = tlx.convert_to_tensor(np.asarray(a_id).reshape(1, 1)).to(device) # a_id是上一个单词的id
            logits, h_state, c_state = net(a_id, h_state, c_state) # 前向传播，更新LSTM的状态
            out = tlx.nn.Softmax(axis=-1)(tlx.transpose(logits, (0,2,1)))
            
            # Without sampling
            # a_id = np.argmax(out[0][0].detach().numpy())
            # Sample from all words, if vocab_size is large,
            # this may have numeric error.
            # a_id = tl.nlp.sample(out[0], diversity)
            # Sample from the top k words.

            a_id = nlp.sample_top(out[0][0].cpu().detach().numpy(), top_k=top_k) # 从top_k个单词中随机选择一个，a_id也更新
            outs_id.append(a_id) # 将生成的单词加入到序列中

        sentence = [vocab.id_to_word(w) for w in outs_id] # 将id转换为单词，构成句子的列表
        sentence = " ".join(sentence)
        # print(diversity, ':', sentence)
        print(top_k, ':', sentence)

print("Save model")
net.save_weights("ch05/ptb_lstm_weights.npz")