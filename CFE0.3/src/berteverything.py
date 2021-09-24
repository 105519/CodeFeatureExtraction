#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import json
import torch
import random
import tokenize
import numpy as np
from tqdm import tqdm
import multiprocessing
from torch.utils.data import DataLoader, Dataset, RandomSampler
from transformers import RobertaConfig, RobertaForMaskedLM, RobertaTokenizer
from transformers import WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup
from parser import remove_comments_and_docstrings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss, MSELoss
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from transformers import RobertaConfig, RobertaForMaskedLM, RobertaTokenizer
from transformers import WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import recall_score, precision_score, f1_score


# In[2]:


class arguments(object):
    def __init__(self):
        pass
args = arguments()


# In[3]:


args.total_length = 512
args.graph_length = 0
args.epochs = 6
args.topic_size = 18

# args.train_batch_size = 1
# args.eval_batch_size = 1

args.gradient_accumulation_steps = 1
args.max_grad_norm = 1.0
args.learning_rate = 5e-5
args.weight_decay = 0.0
args.adam_epsilon = 1e-8

args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args.n_gpu = torch.cuda.device_count()
args.seed = 978438233

def set_seed():
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
set_seed()


# In[4]:


config = RobertaConfig.from_pretrained('microsoft/codebert-base')
config.num_labels = 1
tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base')
model0 = RobertaForMaskedLM.from_pretrained('microsoft/codebert-base', config = config)


# In[5]:


# def get_tokens(code, do_remove):
#     if (do_remove):
#         code = remove_comments_and_docstrings(code, 'python')
#     output_file = open('tmp' + '.py', 'w')
#     print(code, file = output_file)
#     output_file.close()
    
#     tokens = []
#     f = open('tmp' + '.py', 'rb')
#     tokenGenerator = tokenize.tokenize(f.readline)
#     for token in tokenGenerator:
#         if (token.type in [0, 59, 60, 62]): # COMMENT
#             pass
#         elif (token.type in [4, 61]): # NEWLINE
#             pass
#         elif (token.type == 5): # INDENT
#             pass
#         elif (token.type == 6): # DEDENT
#             pass
#         elif (token.type in [1, 2, 3, 54]): # NAME NUMBER STRING OP
#             tokens.append(token.string)
#         else:
#             assert(False)
#     f.close()
#     return tokens

# def search(path):
#     data = []
#     if (os.path.isdir(path)):
#         for filename in os.listdir(path):
#             data.extend(search(path + '/' + filename))
#     else:
#         assert(os.path.isfile(path))
#         input_file = open(path, 'r')
#         code = input_file.read()
#         try:
#             tokens = get_tokens(code, True)
#         except:
#             tokens = get_tokens(code, False)
#         if (len(tokens) != 0):
#             tokens = [tokenizer.tokenize(tokens[0])] \
#                    + [tokenizer.tokenize('@ ' + x)[1 :] for x in tokens[1 :]]
#             tokens = [y for x in tokens for y in x]
#             tokens = tokens[: args.total_length - 2]
            
#             code_ids = [tokenizer.cls_token_id] + tokenizer.convert_tokens_to_ids(tokens)
#             position_ids = [i + tokenizer.pad_token_id + 1 for i in range(len(code_ids))]
#             padding_length = args.total_length - len(code_ids)
#             code_ids += [tokenizer.pad_token_id] * padding_length
#             position_ids += [tokenizer.pad_token_id] * padding_length
            
#             data.append((code_ids, position_ids))
#         input_file.close()
#     return data

# topic_map = dict()

# def read_data(repo_file, topic_file):
#     dataset = []
#     f = open(repo_file, 'r')
#     repos = []
#     for line in f:
#         _, repo = line.strip().split(chr(9))
#         repo = repo[repo.rfind('/', 0, repo.rfind('/') - 1) + 1 :]
#         repos.append(repo)
#     f.close()
#     f = open(topic_file, 'r')
#     topics = json.loads(f.readline())['repos']
#     f.close()
    
#     for x in topics:
#         for y in x:
#             if (y not in topic_map):
#                 topic_map[y] = len(topic_map)
    
#     for repo, topic in tqdm(zip(repos, topics), total = len(repos)):
#         labels = [0] * len(topic_map)
#         for x in topic:
#             labels[topic_map[x]] = 1
#         data = search('../data/py150_files/' + repo)
#         dataset.append(([x for x, y in data], [y for x, y in data], labels))
#     return dataset

# dataset = read_data('../data/py150/github_repos.txt', '../data/py150/repo_topics2.jsonl')
# dataset = [x for x in dataset if (len(x[0]) != 0)]
# print('get', len(dataset), 'datas in total')


# In[6]:


f = open('dataset.jsonl', 'r')
dataset0 = json.loads(f.readline())['dataset']
f.close()

topic_map = dict()

def read_data(repo_file, topic_file):
    f = open(repo_file, 'r')
    repos = []
    for line in f:
        _, repo = line.strip().split(chr(9))
        repo = repo[repo.rfind('/', 0, repo.rfind('/') - 1) + 1 :]
        repos.append(repo)
    f.close()
    f = open(topic_file, 'r')
    topics = json.loads(f.readline())['repos']
    f.close()
    
    for x in topics:
        for y in x:
            if (y not in topic_map):
                topic_map[y] = len(topic_map)
    dataset = dataset0
    for idx, topic in enumerate(topics):
        labels = [0] * len(topic_map)
        for x in topic:
            labels[topic_map[x]] = 1
        dataset[idx].append(labels)
    return dataset

class TextDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples
        
    def __len__(self):
        return len(self.examples)
        
    def __getitem__(self, item):
        data = self.examples[item]
        code_ids, position_ids, labels = data
        return torch.tensor(code_ids), torch.tensor(position_ids), torch.tensor(labels)

dataset = read_data('../data/py150/github_repos.txt', '../data/py150/repo_topics2.jsonl')
dataset = [x for x in dataset if (len(x[0]) != 0)]


# In[7]:


class TextDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples
        
    def __len__(self):
        return len(self.examples)
        
    def __getitem__(self, item):
        data = self.examples[item]
        code_ids, position_ids, labels = data
        return torch.tensor(code_ids), torch.tensor(position_ids), torch.tensor(labels)


# In[8]:


output_file = open('bertresult.txt', 'w')

def evaluate(model):
    logits = []
    y_trues = []
    for data in tqdm(test_dataloader, total = len(test_dataloader)):
        code_ids, position_ids, labels = data
        code_ids = code_ids.to(args.device)
        position_ids = position_ids.to(args.device)
        labels = labels[:, args.topic_num].to(args.device)
        model.eval()
        with torch.no_grad():
            prob = model(code_ids, position_ids)
            logits.append(prob.view(-1).cpu().numpy())
            y_trues.append(labels.view(-1).cpu().numpy())
    logits = np.concatenate(logits, 0)
    y_trues = np.concatenate(y_trues, 0)
    y_preds = logits > 0.5
    TP = FP = TN = FN = 0
    for x, y in zip(y_trues > 0.5, y_preds):
        if (x == 1 and y == 1):
            TP += 1
        if (x == 1 and y == 0):
            FP += 1
        if (x == 0 and y == 0):
            TN += 1
        if (x == 0 and y == 1):
            FN += 1
    print('positive:', TP, '/', FP, ':', round(TP / (TP + FP), 2))
    print('negative:', TN, '/', FN, ':', round(TN / (TN + FN), 2))
    print('f1:', float(f1_score(y_trues, y_preds)))
    print('positive:', TP, '/', FP, ':', round(TP / (TP + FP), 2), file = output_file)
    print('negative:', TN, '/', FN, ':', round(TN / (TN + FN), 2), file = output_file)
    print('f1:', float(f1_score(y_trues, y_preds)), file = output_file)

class Model(nn.Module):   
    def __init__(self, encoder, config):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config = config
        self.rnn = nn.LSTM(config.hidden_size, config.hidden_size, 1)
        
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, 2) # * len(topic_map)
        
    def forward(self, code_ids, position_ids, labels = None):
        code_ids = code_ids.view(-1, args.total_length)
        position_ids = position_ids.view(-1, args.total_length)
        code_embeddings = self.encoder.roberta.embeddings.word_embeddings(code_ids)
        
        h = torch.randn(1, 1, self.config.hidden_size).to(args.device)
        c = torch.randn(1, 1, self.config.hidden_size).to(args.device)
        for i in range(code_embeddings.size(0)):
            bert_output = self.encoder.roberta(inputs_embeds = code_embeddings[i].view(1, args.total_length, -1),
                                               position_ids = position_ids[i].view(1, -1))
            _, (h, c) = self.rnn(bert_output[0][:, 0, :].view(-1, 1, config.hidden_size), (h, c))
        x = h[0]
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x).view(-1, 2)
        prob = F.softmax(x, dim = 1)[:, 1]
        if (labels is None):
            return prob
        else:
            labels = labels.view(-1)
            loss_function = CrossEntropyLoss()
            loss = loss_function(x, labels)
            return loss

def my_train():
    model = Model(model0, config)
    model.to(args.device)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    optimizer = torch.optim.Adam(model.parameters(), lr = args.learning_rate)
    
    for epoch_num in range(args.epochs):
        train_dataset = []
        cnt_positive = 0
        cnt_negative = 0
        for x in dataset[: int(len(dataset) * 0.75)]:
            if (x[2][args.topic_num] == 1):
                train_dataset.append(x)
                cnt_positive += 1
        for x in dataset[: int(len(dataset) * 0.75)]:
            if (x[2][args.topic_num] == 0) and (cnt_negative < 16 * cnt_positive):
                train_dataset.append(x)
                cnt_negative += 1
#         print('P, N =', cnt_positive, cnt_negative)
        
        train_data = TextDataset(train_dataset)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler = train_sampler, drop_last = True, num_workers = 4) #TODO
        
        step = 0
        
        for data in tqdm(train_dataloader, total = len(train_dataloader)):
            code_ids, position_ids, labels = data
            code_ids = code_ids[:, : 12, :]
            position_ids = position_ids[:, : 12, :]
            code_ids = code_ids.to(args.device)
            position_ids = position_ids.to(args.device)
            labels = labels[:, args.topic_num].view(-1, 1).to(args.device)
            model.train()
            loss = model(code_ids, position_ids, labels)
#             if args.n_gpu > 1: # TODO
#                 loss = loss.mean()
            if (labels[0, 0] == 1): # TODO
                loss = loss / cnt_positive * (cnt_positive + cnt_negative)
            else:
                loss = loss / cnt_negative * (cnt_positive + cnt_negative)
            loss.backward()
            step += 1
            if (step % 32 == 0):
                optimizer.step()
                optimizer.zero_grad()
        if (step % 32 != 0):
            optimizer.step()
            optimizer.zero_grad()
        evaluate(model)

for i in range(7, args.topic_size):
    for x in topic_map:
        if (topic_map[x] == i):
            print('\n[TOPIC]:', x, i)
            print('\n[TOPIC]:', x, file = output_file)
    args.topic_num = i
    random.shuffle(dataset)
    test_dataset = dataset[int(len(dataset) * 0.75) :]
    test_data = TextDataset(test_dataset)
    test_sampler = RandomSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler = test_sampler, drop_last = False, num_workers = 4) #TODO
    my_train()

output_file.close()


# In[ ]:


torch.cuda.empty_cache()

