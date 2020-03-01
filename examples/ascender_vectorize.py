import os
import math
import re
from time import time
from multiprocessing import Pool
import requests
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from bert_serving.client import BertClient

def norm_vec(v):
    n = np.linalg.norm(v)
    if n != 0.0:
        return v / n
    else:
        return v


def norm_vecs(vecs):
    return [norm_vec(v) for v in vecs]


def nearest(target, vecs):
    # Wouldn't it be better if this was cosine?
    # However, I am getting good results with this.
    scores = [np.linalg.norm(target - v) for v in vecs]
    scores = list(zip(scores, list(range(len(scores)))))
    scores.sort(key=lambda x: x[0])
    return scores


def nearest_cos(target, vecs):
    # Wouldn't it be better if this was cosine?
    # However, I am getting good results with this.
    if len(vecs) == 0:
        return []
    target = np.reshape(target, (1, -1))
    vecs = np.stack(vecs)
    scores = cosine_similarity(target, vecs)
    scores = scores.tolist()[0]
    return scores
    # scores = list(zip(scores, list(range(len(scores)))))
    # scores.sort(key=lambda x: -x[0])
    # return scores


def word2vec(words, precision=None):
    res = requests.post("http://localhost:5000/word2vec", json={
        'words': words,
        'precision': precision,
    })
    res.raise_for_status()
    dictionary = res.json()
    npdict = {}
    for word, vec in dictionary.items():
        if vec is not None:
            npdict[word] = np.array(vec, dtype=np.float32)
    return npdict


def tokenize(text):
    # Convert newlines to whitespace
    text = text.replace('\n', ' ')
    # text = text.replace("'ll", ' will')
    # Normalize sentence punctuation
    text = text.replace('?', '.').replace('!', '.').replace(',', ' ')
    # Filter everything except alpha-numeric, spaces, and periods
    text = re.sub('[^A-Za-z0-9 .]', '', text)
    text = text.replace('.', ' ')
    text = re.sub(' +', ' ', text)
    words = text.split(' ')
    # Remove empty strings
    words = [w for w in words if len(w) > 0]
    return words


def calc_idf(para):
    f = {}

    uniq_words = set(para)
    for word in uniq_words:
        c = f.setdefault(word, 0)
        c += 1
        f[word] = c
    f2 = {}
    for k, v in f.items():
        v = math.log(len(para) / (1.0 + v))
        f2[k] = v
    return f2


def calc_tfidf(p, idf):
    """Returns a dictionary of words present in p with their tfidf scores"""
    counts = {}
    for word in p:
        c = counts.setdefault(word, 0)
        c += 1
        counts[word] = c
    tfidf = {}
    for k, c in counts.items():
        tfidf[k] = (c / len(p)) * idf[k]
    return tfidf


def tfidf_vector(words, model, idf):
    tfidf = calc_tfidf(words, idf)
    acc = np.zeros((300,))
    count = 0
    oov = set([])
    for word in words:
        try:
            v = model[word]
            acc += v * tfidf[word]
            count += 1
        except KeyError:
            oov.add(word)
    # acc = acc / np.linalg.norm(acc)
    if count > 0:
        acc = acc / count
    acc = norm_vec(acc)
    # print('OOV: ', oov)
    return acc


def snippets2tokens(snippets):
    return [tokenize(s) for s in snippets]


def word2vec_local(words, model):
    t0 = time()
    dictionary = {}
    for word in words:
        try:
            v = model[word].tolist()
            dictionary[word] = v
        except Exception as e:
            dictionary[word] = None
    npdict = {}
    for word, vec in dictionary.items():
        if vec is not None:
            npdict[word] = np.array(vec, dtype=np.float32)
    #print('word2vec lookup time %0.2f sec' % (time() - t0))
    return npdict


def calc_dict_idf(tokenized, model):
    # Compute combined dictionary and TF-IDF across all documents
    t1 = time()
    all_snippets = []
    for doc in tokenized:
        all_snippets.extend(doc)
    idf = calc_idf(all_snippets)
    words = list(idf.keys())
    #print('calc_idf and words for %i snippets in %0.2f secs' %
    #      (len(all_snippets), time() - t1))

    t1 = time()
    if model is None:
        dictionary = word2vec(words, precision=4)
    else:
        dictionary = word2vec_local(words, model)
    print('word2vec of %d words in %0.2f secs' % (len(words), time() - t1))
    return dictionary, idf




def sents2vec(sents, model=None):
    """Tokenize all snippets"""
    tokenized = []

    for sent in sents:
        tokenized.append(tokenize(sent))

    dictionary, idf = calc_dict_idf(tokenized, model)

    t1 = time()
    vectors =   []

    for snip in tokenized:
        vectors.append(tfidf_vector(snip, dictionary, idf))

    return vectors
def sents_vectorize_bert(sents):
    bc = BertClient(ip='192.168.1.16')  # ip address of the GPU machine

    vectors = bc.encode(sents)

    return vectors
