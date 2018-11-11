import re
from itertools import chain
import pandas as pd
import numpy as np
import pickle


text = open('./demo.txt', encoding='utf-8').read()
""" 原始数据:

text: '人/b  们/e  常/s  说/s  生/b  活/e  是/s  一/s  部/s  教/b  科/m  书/e  ，/s  而/s  血/s  与/s  火/s  的/s  战/b  争/e  更/s  是/s  不/b'
"""

sentences = re.split('[，。！？、‘’“”]/[bems]', text) 
""" 从标点处断开句子:

[
    '人/b 们/e ...',
    '而/s 血/s ...'
]
"""

sentences = list(filter(lambda x: x.strip(), sentences)) # 筛选掉为空的项
sentences = list(map(lambda x: x.strip(), sentences)) # 删除空格

# To numpy array
words, labels = [], []
print('Start creating words and labels...')
for sentence in sentences:
    ''' sentence:
    '人/b  们/e  常/s  说/s  生/b  活/e  是/s  一/s  部/s  教/b  科/m  书/e '
    '''
    groups = re.findall('(.)/(.)', sentence)
    ''' groups:
    [('人', 'b'), ('们', 'e'), ...]
    '''
    arrays = np.asarray(groups) # asarray 干嘛呢， arrray就好了嘛
    ''' arrays:
    [['人' 'b']
     ['们' 'e']
     ['常' 's']
     ['说' 's']
     ['生' 'b']]
    '''
    words.append(arrays[:, 0])
    labels.append(arrays[:, 1])
print('Words Length', len(words), 'Labels Length', len(labels))
print('Words Example', words[0])
print('Labels Example', labels[0])
''' 把字和标注分开
words:
[
    array(['人', '们', '常', '说', '生', '活', '是', '一', '部', '教', '科', '书'],dtype='<U1'), 
    array(['而', '血', '与', '火', '的', '战', '争', '更', '是', '不'], dtype='<U1')
]
labels:
[
    array(['b', 'e', 's', 's', 'b', 'e', 's', 's', 's', 'b', 'm', 'e'],dtype='<U1'), 
    array(['s', 's', 's', 's', 's', 'b', 'e', 's', 's', 'b'], dtype='<U1')
]
'''
# Merge all words
all_words = list(chain(*words))
'''
['人', '们', '常', '说', '生', '活', '是', '一', '部', '教', '科', '书', '而', '血', '与', '火', '的', '战', '争', '更', '是', '不']
'''
# All words to Series
all_words_sr = pd.Series(all_words)
'''
list变成panda Series
0     人
1     们
2     常
'''
# Get value count, index changed to set
all_words_counts = all_words_sr.value_counts()
'''
计算每个字出现的频率,得到另一个Series
是    2
不    1
更    1
------------------------------------------------------
关键在此，value_counts之后得到的新Series的index变成了 中文字
------------------------------------------------------
'''
# Get words set
all_words_set = all_words_counts.index # '''把Index object拿出来'''
# Get words ids
all_words_ids = range(1, len(all_words_set) + 1)

# Dict to transform
word2id = pd.Series(all_words_ids, index=all_words_set)
id2word = pd.Series(all_words_set, index=all_words_ids)
'''中文字 和 数字 互相index
word2id
是     1
不     2
更     3
科     4
人     5

id2word
1     是
2     不
3     更
4     科
5     人
'''

# Tag set and ids
tags_set = ['x', 's', 'b', 'm', 'e']
tags_ids = range(len(tags_set))

# Dict to transform
tag2id = pd.Series(tags_ids, index=tags_set)
id2tag = pd.Series(tags_set, index=tag2id)

max_length = 32

def x_transform(words):
    ids = list(word2id[words])
    ''' 把 中文字序列 转换成 数字序列
    words是一个数组，得到的ids也是一个数组
    '''
    if len(ids) >= max_length:
        ids = ids[:max_length]
    # 大于max_length就截断
    ids.extend([0] * (max_length - len(ids)))
    # 不够max_length的补0, extend是在原list上加，并不返回
    return ids


def y_transform(tags):
    ids = list(tag2id[tags])
    if len(ids) >= max_length:
        ids = ids[:max_length]
    ids.extend([0] * (max_length - len(ids)))
    return ids


print('Starting transform...')
'''
words:
[
    array(['人', '们', '常', '说', '生', '活', '是', '一', '部', '教', '科', '书'],dtype='<U1'), 
    array(['而', '血', '与', '火', '的', '战', '争', '更', '是', '不'], dtype='<U1')
]
labels:
[
    array(['b', 'e', 's', 's', 'b', 'e', 's', 's', 's', 'b', 'm', 'e'],dtype='<U1'), 
    array(['s', 's', 's', 's', 's', 'b', 'e', 's', 's', 'b'], dtype='<U1')
]
'''
data_x = list(map(lambda x: x_transform(x), words))
data_y = list(map(lambda y: y_transform(y), labels))
'''
data_x
[
    [5, 12, 16, 11, 20, 14, 1, 21, 6, 17, 4, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
    [9, 18, 8, 10, 15, 13, 19, 3, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
]
等于把中文句子中的字都变成了数字
data_x是一整个库的句子，很多条
库是由很多句子组成的
'''
print('Data X Length', len(data_x), 'Data Y Length', len(data_y))
print('Data X Example', data_x[0])
print('Data Y Example', data_y[0])

data_x = np.asarray(data_x)
data_y = np.asarray(data_y)

from os import makedirs
from os.path import exists, join

path = 'data/'

if not exists(path):
    makedirs(path)

print('Starting pickle to file...')
with open(join(path, 'data.pkl'), 'wb') as f:
    pickle.dump(data_x, f)
    pickle.dump(data_y, f)
    pickle.dump(word2id, f)
    pickle.dump(id2word, f)
    pickle.dump(tag2id, f)
    pickle.dump(id2tag, f)
print('Pickle finished')
