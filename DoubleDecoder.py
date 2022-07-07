import os
import sys
import math

from collections import Counter
import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

def load_data(in_file,output_file,info_file):
    en,cn,info = [],[],[]
    with open(in_file, 'r') as f:
        for line in f:
            line_list = line.split()
            en.append(["BOS"]+line_list+["EOS"])
    with open(output_file, 'r') as f:
        for line in f:
            line_list = line.split()
            cn.append(["BOS"]+line_list+["EOS"])
    with open(info_file, 'r') as f:
        for line in f:
            line_list = line.split()
            info.append(["BOS"] + line_list + ["EOS"])
    return en, cn,info
train_srcfile = '/home/ahf/Bug_fix_info/data/train_output/new_buggy.txt'
train_tgtfile = "/home/ahf/Bug_fix_info/data/train_output/original_fix.txt"
train_infofile = "/home/ahf/Bug_fix_info/data/train_output/info.txt"
dev_srcfile = '/home/ahf/Bug_fix_info/data/val_output/new_buggy.txt'
dev_tgtfile = "/home/ahf/Bug_fix_info/data/val_output/original_fix.txt"
dev_infofile = "/home/ahf/Bug_fix_info/data/val_output/info.txt"
train_en, train_cn,train_info = load_data(train_srcfile,train_tgtfile,train_infofile)
dev_en, dev_cn,dev_info = load_data(dev_srcfile,dev_tgtfile,dev_infofile)

UNK_IDX = 0
PAD_IDX = 1
def build_dict(sentences, max_words=50000):
    word_count = Counter()
    for sentence in sentences:
        for s in sentence:
            word_count[s] += 1
    ls = word_count.most_common(max_words)
    total_words = len(ls) + 2    # 两个特殊的字符UNK和PAD
    word_dict = {w[0]: index+2 for index, w in enumerate(ls)}   # 字典的前两个位置放特殊字符
    word_dict['UNK'] = UNK_IDX
    word_dict['PAD'] = PAD_IDX
    return word_dict, total_words

en_dict, en_total_words = build_dict(train_en)
cn_dict, cn_total_words = build_dict(train_cn)
info_dict,info_total_words = build_dict(train_info)
inv_en_dict = {v:k for k, v in en_dict.items()}
inv_cn_dict = {v:k for k, v in cn_dict.items()}
inv_info_dict = {v:k for k, v in info_dict.items()}

def encode(en_sentences, cn_sentences,info_sentences, en_dict, cn_dict,info_dict, sort_by_len=True):
    length = len(en_sentences)
    out_en_sentences = [[en_dict.get(w, 0) for w in sent] for sent in en_sentences]
    out_cn_sentences = [[cn_dict.get(w, 0) for w in sent] for sent in cn_sentences]
    out_info_sentences = [[info_dict.get(w, 0) for w in sent] for sent in info_sentences]
    # 根据英语句子的长度排序
    def len_argsort(seq):  # 这个seq是一个二维矩阵， 每一行是一个句子， 且都已经用单词在字典中的位置进行了编码
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    if sort_by_len:
        sorted_index = len_argsort(out_en_sentences)
        out_en_sentences = [out_en_sentences[i] for i in sorted_index]
        out_cn_sentences = [out_cn_sentences[i] for i in sorted_index]
        out_info_sentences = [out_info_sentences[i] for i in sorted_index]
    return out_en_sentences, out_cn_sentences,out_info_sentences


train_en, train_cn,train_info = encode(train_en, train_cn,train_info, en_dict, cn_dict,info_dict)
dev_en, dev_cn,dev_info = encode(dev_en, dev_cn, dev_info,en_dict, cn_dict,info_dict)


# 这个函数的作用是我们输入训练集的样本个数， batch_size大小， 就会返回多批 连续的batch_size个索引， 每一个索引代表一个样本
# 也就是可以根据这个索引去拿到一个个的batch
def get_minibatches(n, minibatch_size, shuffle=True):
    idx_list = np.arange(0, n, minibatch_size)
    if shuffle:
        np.random.shuffle(idx_list)
    minibatches = []
    for idx in idx_list:
        minibatches.append(np.arange(idx, min(idx + minibatch_size, n)))
    return minibatches  # 这个会返回多批连着的bath_size个索引


# get_minibatches(len(train_en), 32)

# 这个函数是在做数据预处理， 由于每个句子都不是一样长， 所以通过这个函数就可以把句子进行补齐， 不够长的在句子后面添加0
def prepare_data(seqs):
    lengths = [len(seq) for seq in seqs]  # 得到每个句子的长度
    n_samples = len(seqs)  # 得到一共有多少个句子
    max_len = np.max(lengths)  # 找出最大的句子长度

    x = np.zeros((n_samples, max_len)).astype('int32')  # 按照最大句子长度生成全0矩阵
    x_lengths = np.array(lengths).astype('int32')
    for idx, seq in enumerate(seqs):  # 把有句子的位置填充进去
        x[idx, :lengths[idx]] = seq
    return x, x_lengths  # x_mask


def gen_examples(en_sentences, cn_sentences,info_sentences, batch_size):
    minibatches = get_minibatches(len(en_sentences), batch_size)  # 得到batch个索引
    all_ex = []
    for minibatch in minibatches:  # 每批数据的索引
        mb_en_sentences = [en_sentences[t] for t in minibatch]  # 取数据
        mb_cn_sentences = [cn_sentences[t] for t in minibatch]  # 取数据
        mb_info_sentences = [info_sentences[t] for t in minibatch]
        mb_x, mb_x_len = prepare_data(mb_en_sentences)  # 填充成一样的长度， 但是要记录一下句子的真实长度， 这个在后面输入网络的时候得用
        mb_y, mb_y_len = prepare_data(mb_cn_sentences)
        mb_z, mb_z_len = prepare_data(mb_info_sentences)
        all_ex.append((mb_x, mb_x_len, mb_y, mb_y_len,mb_z, mb_z_len))
    return all_ex


batch_size = 4
train_data = gen_examples(train_en, train_cn,train_info, batch_size)  # 产生训练集
random.shuffle(train_data)
dev_data = gen_examples(dev_en, dev_cn,dev_info, batch_size)  # 产生验证集

print(train_data[1][0].shape, train_data[1][1].shape, train_data[1][2].shape, train_data[1][3].shape)


class PlainEncoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, dropout=0.2):
        super(PlainEncoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size, batch_first=True)  # 这个地方其实写的有些不太号理解， 第一个维度应该是embed_size，这里为了方便，相等了
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, lengths):
        # 这里需要输入lengths， 因为每个句子是不一样长的，我们需要每个句子最后一个时间步的隐藏状态,所以需要知道句子有多长， x表示一个batch里面的句子

        # 把batch里面的seq按照长度排序
        sorted_len, sorted_idx = lengths.sort(0, descending=True)  # d_le sorted_len表示排好序的数组， sorted_index表示每个元素在原数组位置
        x_sorted = x[sorted_idx.long()]  # 句子已经按照seq长度排好序
        embedded = self.dropout(self.embed(x_sorted))  # [batch_size, seq_len, embed_size]

        # 下面一段代码处理变长序列
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, sorted_len.long().cpu().data.numpy(),
                                                            batch_first=True)  # 这里的data.numpy()是原来张量的克隆， 然后转成了numpy数组， 相当于clone().numpy()
        # 上面这句话之后， 会把变长序列的0都给去掉， 之前填充的字符都给压扁
        packed_out, hid = self.rnn(packed_embedded)  # 通过这句话就可以得到batch中每个样本的真实隐藏状态
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)  # 这里是在填充回去， 看下面的例子就懂了
        _, original_idx = sorted_idx.sort(0, descending=False)  # 这里是为了还是让短的句子在前面
        out = out[original_idx.long()].contiguous()  # contiguous是为了把不连续的内存单元连续起来
        hid = hid[:, original_idx.long()].contiguous()

        return out, hid[[-1]]  # 把最后一层的hid给拿出来  这个具体看上面的简单演示


# 这个基本上和Encoder是一致的， 无非就是初始化的h换成了Encoder之后的h
class PlainDecoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, dropout=0.2):
        super(PlainDecoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, y, y_lengths, hid):
        # y: [batch_size, seq_len-1]
        sorted_len, sorted_idx = y_lengths.sort(0, descending=True)  # 依然是句子从长到短排序
        y_sorted = y[sorted_idx.long()]
        hid = hid[:, sorted_idx.long()]

        y_sorted = self.dropout(self.embed(y_sorted))  # [batch_size, outpout_length, embed_size]

        pack_seq = nn.utils.rnn.pack_padded_sequence(y_sorted, sorted_len.long().cpu().data.numpy(), batch_first=True)
        out, hid = self.rnn(pack_seq, hid)  # 这个计算的是每个有效时间步单词的最后一层的隐藏状态
        unpacked, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)  # [batch, seq_len-1, hidden_size]
        _, original_idx = sorted_idx.sort(0, descending=False)
        output_seq = unpacked[original_idx.long()].contiguous()  # [batch, seq_len-1, hidden_size]

        hid = hid[:, original_idx.long()].contiguous()  # [1， batch, hidden_size]
        output = F.log_softmax(self.out(output_seq), -1)
        # [batch, seq_len-1, vocab_size]   表示每个样本每个时间不长都有一个vocab_size的维度长度， 表示每个单词的概率

        return output, hid


class PlainSeq2Seq(nn.Module):
    def __init__(self, encoder, code_decoder,info_decoder):
        super(PlainSeq2Seq, self).__init__()
        self.encoder = encoder
        self.code_decoder = code_decoder
        self.info_decoder = info_decoder
    def forward(self, x, x_lengths, y, y_lengths,z,z_lengths):
        encoder_out, hid = self.encoder(x, x_lengths)  # encoder进行编码
        code_output, code_hid = self.code_decoder(y, y_lengths, hid)  # deocder 负责解码
        info_output,info_hid = self.info_decoder(z,z_lengths,hid)
        return code_output, info_output

    def translate(self, x, x_lengths, y,z, max_length=10):  # 这个是进来一个句子进行翻译  max_length句子的最大长度
        encoder_out, hid = self.encoder(x, x_lengths)  # 解码
        code_preds,info_preds = [],[]
        batch_size = x.shape[0]
        attns = []
        for i in range(max_length):
            code_output, code_hid = self.code_decoder(y, torch.ones(batch_size).long().to(y.device), hid=hid)
            info_output, info_hid = self.info_decoder(y, torch.ones(batch_size).long().to(z.device), hid=hid)
            y = code_output.max(2)[1].view(batch_size, 1)
            z = info_output.max(2)[1].view(batch_size, 1)
            code_preds.append(y)
            info_preds.append(z)
        return torch.cat(code_preds, 1), torch.cat(info_preds, 1)


# masked cross entropy loss
class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, input, target, mask):
        # input: [batch_size, seq_len, vocab_size]    每个单词的可能性
        input = input.contiguous().view(-1, input.size(2))  # [batch_size*seq_len-1, vocab_size]
        target = target.contiguous().view(-1, 1)  # [batch_size*seq_len-1, 1]

        mask = mask.contiguous().view(-1, 1)  # [batch_size*seq_len-1, 1]
        output = -input.gather(1, target) * mask  # 在每个vocab_size维度取正确单词的索引， 但是里面有很多是填充进去的， 所以mask去掉这些填充的
        # 这个其实在写一个NLloss ， 也就是sortmax的取负号
        output = torch.sum(output) / torch.sum(mask)

        return output  # [batch_size*seq_len-1, 1]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dropout = 0.2
hidden_size = 100
encoder = PlainEncoder(vocab_size=en_total_words, hidden_size=hidden_size, dropout=dropout)
code_decoder = PlainDecoder(vocab_size=cn_total_words, hidden_size=hidden_size, dropout=dropout)
info_decoder = PlainDecoder(vocab_size=info_total_words, hidden_size=hidden_size, dropout=dropout)

model = PlainSeq2Seq(encoder, code_decoder,info_decoder)
model = model.to(device)
loss_fn = LanguageModelCriterion().to(device)
optimizer = torch.optim.Adam(model.parameters())


# 定义训练和验证函数
def evaluate(model, data):
    model.eval()
    total_num_words = total_loss = 0.
    with torch.no_grad():
        for it, (mb_x, mb_x_len, mb_y, mb_y_len,mb_z,mb_z_len) in enumerate(data):
            mb_x = torch.from_numpy(mb_x).to(device).long()  # 这个是一个batch的英文句子 大小是[batch_size, seq_len]
            mb_x_len = torch.from_numpy(mb_x_len).to(device).long()  # 每个句子的长度
            mb_code_input = torch.from_numpy(mb_y[:, :-1]).to(device).long()  # 解码器那边的输入， 输入一个单词去预测另外一个单词
            mb_code_output = torch.from_numpy(mb_y[:, 1:]).to(device).long()  # 解码器那边的输出  [batch_size, seq_len-1]
            mb_info_input = torch.from_numpy(mb_z[:, :-1]).to(device).long()  # 解码器那边的输入， 输入一个单词去预测另外一个单词
            mb_info_output = torch.from_numpy(mb_z[:, 1:]).to(device).long()  # 解码器那边的输出  [batch_size, seq_len-1]
            mb_y_len = torch.from_numpy(mb_y_len - 1).to(device).long()  # 这个减去1， 因为没有了最后一个  [batch_size, seq_len-1]
            mb_z_len = torch.from_numpy(mb_z_len - 1).to(device).long()  # 这个减去1， 因为没有了最后一个  [batch_size, seq_len-1]
            mb_y_len[mb_y_len <= 0] = 1  # 这句话是为了以防出错
            mb_z_len[mb_z_len <= 0] = 1  # 这句话是为了以防出错

            mb_code_pred, mb_info_pred = model(mb_x, mb_x_len, mb_code_input, mb_y_len,mb_info_input,mb_z_len)

            mb_code_out_mask = torch.arange(mb_y_len.max().item(), device=device)[None, :] < mb_y_len[:, None]
            mb_info_out_mask = torch.arange(mb_z_len.max().item(), device=device)[None, :] < mb_z_len[:, None]

            # [batch_size, mb_y_len.max()], 上面是bool类型， 下面是float类型， 只计算每个句子的有效部分， 填充的那部分去掉
            mb_code_out_mask = mb_code_out_mask.float()  # [batch_size, seq_len-1]  因为mb_y_len.max()就是seq_len-1
            mb_info_out_mask = mb_info_out_mask.float()  # [batch_size, seq_len-1]  因为mb_y_len.max()就是seq_len-1

            code_loss = loss_fn(mb_code_pred, mb_code_output, mb_code_out_mask)
            info_loss = loss_fn(mb_info_pred, mb_info_output, mb_info_out_mask)
            num_code_words = torch.sum(mb_y_len).item()
            num_info_words = torch.sum(mb_z_len).item()
            total_loss += code_loss.item() * num_code_words+info_loss.item()*num_info_words
            total_num_words += num_code_words+num_info_words
    print('Evaluation loss', total_loss / total_num_words)


def train(model, data, num_epochs=20):
    for epoch in range(num_epochs):
        model.train()
        total_num_words = total_loss = 0.
        for it, (mb_x, mb_x_len, mb_y, mb_y_len, mb_z, mb_z_len) in enumerate(data):
            mb_x = torch.from_numpy(mb_x).to(device).long()  # 这个是一个batch的英文句子 大小是[batch_size, seq_len]
            mb_x_len = torch.from_numpy(mb_x_len).to(device).long()  # 每个句子的长度
            mb_code_input = torch.from_numpy(mb_y[:, :-1]).to(device).long()  # 解码器那边的输入， 输入一个单词去预测另外一个单词
            mb_code_output = torch.from_numpy(mb_y[:, 1:]).to(device).long()  # 解码器那边的输出  [batch_size, seq_len-1]
            mb_info_input = torch.from_numpy(mb_z[:, :-1]).to(device).long()  # 解码器那边的输入， 输入一个单词去预测另外一个单词
            mb_info_output = torch.from_numpy(mb_z[:, 1:]).to(device).long()  # 解码器那边的输出  [batch_size, seq_len-1]
            mb_y_len = torch.from_numpy(mb_y_len - 1).to(device).long()  # 这个减去1， 因为没有了最后一个  [batch_size, seq_len-1]
            mb_z_len = torch.from_numpy(mb_z_len - 1).to(device).long()  # 这个减去1， 因为没有了最后一个  [batch_size, seq_len-1]
            mb_y_len[mb_y_len <= 0] = 1  # 这句话是为了以防出错
            mb_z_len[mb_z_len <= 0] = 1  # 这句话是为了以防出错

            mb_code_pred, mb_info_pred = model(mb_x, mb_x_len, mb_code_input, mb_y_len, mb_info_input, mb_z_len)

            mb_code_out_mask = torch.arange(mb_y_len.max().item(), device=device)[None, :] < mb_y_len[:, None]
            mb_info_out_mask = torch.arange(mb_z_len.max().item(), device=device)[None, :] < mb_z_len[:, None]

            # [batch_size, mb_y_len.max()], 上面是bool类型， 下面是float类型， 只计算每个句子的有效部分， 填充的那部分去掉
            mb_code_out_mask = mb_code_out_mask.float()  # [batch_size, seq_len-1]  因为mb_y_len.max()就是seq_len-1
            mb_info_out_mask = mb_info_out_mask.float()  # [batch_size, seq_len-1]  因为mb_y_len.max()就是seq_len-1

            code_loss = loss_fn(mb_code_pred, mb_code_output, mb_code_out_mask)
            info_loss = loss_fn(mb_info_pred, mb_info_output, mb_info_out_mask)
            num_code_words = torch.sum(mb_y_len).item()
            num_info_words = torch.sum(mb_z_len).item()
            total_loss += code_loss.item() * num_code_words + info_loss.item() * num_info_words
            total_num_words += num_code_words + num_info_words
            loss = code_loss+info_loss
            # 更新
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)  # 这里防止梯度爆炸， 这是和以往不太一样的地方
            optimizer.step()

            if it % 100 == 0:
                print('Epoch', epoch, 'iteration', it, 'loss', loss.item())

        print('Epoch', epoch, 'Training loss', total_loss / total_num_words)
        if epoch % 5 == 0:
            evaluate(model, dev_data)


# 训练
train(model, train_data, num_epochs=20)
