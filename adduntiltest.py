from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
import os
import msgpack
from rich.console import Console
console = Console()
from rich.progress import track
import time
import numpy as np
from src.utils.vocab import Vocab, Indexer
from src.modules.embedding import Embedding

from stanfordcorenlp import StanfordCoreNLP
import numpy as np
import random
from collections import Counter

# glove_input_file = '/home/lcy/adarea/simplepytorch/resources/glove.840B.300d.txt'
# 2196017 300
# word2vec_output_file = '/home/lcy/adarea/simplepytorch/resources/glove.840B.300d.word2vec.txt'
# word2vec_output_file = '/home/lcy/adarea/simplepytorch/models/snli/embedding.msgpack'
#
#
# # (count, dimensions) = glove2word2vec(glove_input_file, word2vec_output_file)
# # print(count, '\n', dimensions)
# #
# # 加载模型
# glove_model = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)
# # 如果希望直接获取某个单词的向量表示，直接以下标方式访问即可
# cat_vec = glove_model['the']
# console.print(cat_vec)
# # 获得单词an的最相似向量的词汇
# console.print(glove_model.most_similar('an'))



# w2id = {}
# id2w = {}
#
#
# def add_symbol(symbol):
#     if symbol not in w2id:
#         id2w[len(id2w)] = symbol
#         w2id[symbol] = len(w2id)
#
# word_list = cls()
# for i in track(range(10), description='vocab load...'):
#     with open(word_file,'rb') as f:
#          # word_list = f.read().split('\n')
#          for line in f:
#              symbol = line.rstrip()
#              word_list.add_symbol(symbol)
#     time.sleep(1)

def get_word_by_candidate_index(find_synonym_index):

    output_dir = '/home/lcy/adarea/simplepytorch/models/snli'
    embedding_file = os.path.join(output_dir, 'embedding.msgpack')
    word_file = os.path.join(output_dir, 'vocab.txt')
    
    for i in track(range(10), description='embedding load...'):
        with open(embedding_file, 'rb') as f:
             embeddings = msgpack.load(f)
        time.sleep(1)
    
    for i in track(range(10), description='vocab load...'):
       vocab = Vocab.load(word_file) 
       time.sleep(1)
    
     
    console.print('输出the的序号',vocab.index('the')) 
    console.print('输出序号3对应的单词',vocab.__getitem__(3))  
    # console.print(embeddings[4:5])
    # console.print(len(embeddings))
    
    #
    top_k = 30
    
    similarity_socre_array = []
    # find_synonym_index = 3868
    
    for (index, word_embedding) in enumerate(embeddings):
        cos_sim = np.array(embeddings[find_synonym_index]).dot(np.array(embeddings[index])) / np.linalg.norm(np.array(embeddings[find_synonym_index])) * np.linalg.norm(np.array(embeddings[index]))
        similarity_socre_array.append(cos_sim)
      
    for i in track(range(10), description='canculate similarity...'):
        time.sleep(1)
    
    print("similarity的相似矩阵",similarity_socre_array)
    console.print("similarity的相似矩阵大小",len(similarity_socre_array))
    top_k_word = []
    
    
    top_k_idx=np.array(similarity_socre_array).argsort()[::-1][0:top_k]
    for index_in_top_k_idx in top_k_idx:
        top_k_word.append(vocab.__getitem__(int(index_in_top_k_idx)))  
    
    
    console.print("前三十个相似单词的index")
    print(top_k_idx) 
    console.print("前三十个相似单词的word")
    print(top_k_word) 
     
    return top_k_idx,top_k_word



########################################################初次裁剪################################################################################

def sentence_prune(test_file_path, test_sentence_index):  
    word_candidate_index_list = []
    file = open(test_file_path, 'r', encoding='utf-8')
    sentences = []
    for line in file:
        input_batch = line.split('\t')[0]
        sentences.append(input_batch)
    file.close()
      
    sentences = sentences[test_sentence_index]
    nlp=StanfordCoreNLP(r'/home/lcy/adsample/NCR2Code/')
    fin = sentences
    console.print("初次剪枝字符串",fin)
    fner=open('nerCops_sentence.txt','w',encoding='utf8')
    ftag=open('pos_tag_sentence.txt','w',encoding='utf8')
    word_dict=[]
    word_tag =[]
    word_candidate_list = []
    word_candidate_index_list = []
    fner.write(" ".join([each[0]+"/"+each[1] for each in nlp.ner(fin) if len(each)==2 ])+"\n")
    ftag.write(" ".join([each[0]+"/"+each[1] for each in nlp.pos_tag(fin) if len(each)==2 ]) +"\n")
    for each in nlp.ner(fin):
        string_word = each
        word_dict.append(string_word[0])
    for each in nlp.pos_tag(fin):
        word_tag.append(each[1])
    for m,n,word_inex in zip(word_dict,word_tag,range(0,len(word_dict))):
        if n == 'VB' or n == 'VBD' or n == 'VBG'  or n == 'VBN' or n == 'VBP' or n == 'VBZ' or n == 'NN' or n == 'NNS' or n == 'NNP' or n == 'JJS' or n == 'JJR' or n == 'JJ'  or n == 'RB' or n == 'RBS' or n == 'RBR':
            word_candidate_list.append(m)
            word_candidate_index_list.append(word_inex)
    return word_candidate_list,word_candidate_index_list






########################################################分词裁剪################################################################################
def listToString(s):
 
    # initialize an empty string
    str1 = ""
    
    # traverse in the string
    for ele in s:
        str1 +=" "+ ele
 
    # return string
    return str1



def record_input_dataset(_path_to_file):
    file = open(_path_to_file, 'r', encoding='utf-8')
    sentences = []
    count = 0 
    for line in file:
        input_batch = line.split('\t')[0]
        sentences.append(input_batch)
        count = count + 1
        #print(input_batch)
    file.close()
    str = '\n'
    f=open("record_token.txt","w")
    f.write(str.join(sentences))
    f.close()
    return  count
    
def word_prune(input_list):    

    nlp=StanfordCoreNLP(r'/home/lcy/adsample/NCR2Code/')
    
    # fin=open('traintest.txt','r',encoding='utf8')
    fin = listToString(input_list)
    console.print("合成字符串",fin)
    fner=open('nerCops.txt','w',encoding='utf8')
    ftag=open('pos_tag.txt','w',encoding='utf8')
    
    word_dict=[]
    word_tag =[]
    

    fner.write(" ".join([each[0]+"/"+each[1] for each in nlp.ner(fin) if len(each)==2 ])+"\n")
    ftag.write(" ".join([each[0]+"/"+each[1] for each in nlp.pos_tag(fin) if len(each)==2 ]) +"\n")
                #word_dict.append(i)
                #word_tag.append(i)
    for each in nlp.ner(fin):
        string_word = each
        word_dict.append(string_word[0])
    for each in nlp.pos_tag(fin):
        word_tag.append(each[1])
    n_standard =  word_tag[0]           
    m_standard =  word_dict[0]

    console.print("单词字典",word_dict)
    console.print("单词标记",word_tag)    
    console.print("参考标签",n_standard)
    console.print("参考单词",m_standard)
    
    NN_word_tag = []
                
    for m,n in zip(word_dict,word_tag):
        if n == 'NN' or n == 'NNS' or n == 'NNP':
           NN_word_tag.append(m)
    
    fner.close()
    ftag.close()
    
    return NN_word_tag


def adsample_comput_important(word_tag): 
        count = 0
        for n in word_tag:
            if n == 'NN' or n == 'NNS' or n == 'NNP':
               count = count + 1;
        print("显著性",count/len(word_tag))
        return count/len(word_tag)

#########################################返回一个对抗文本（根据所选单词）################################################################

def get_an_optimization_adsample(prior_sentence, candidate_index, word_synonym):
    prior_sentence[candidate_index]=word_synonym
    return prior_sentence


#########################################test################################################################



# console.print("单词剪枝完毕")
# prune_list = word_prune(top_k_word)
# console.print("单词列表",prune_list)
# console.print("单词列表长度",len(prune_list))
word_candidate_list = []
word_candidate_index_list = []
test_file_path = "/home/lcy/adarea/simplepytorch/data/snli/test.txt"
word_candidate_list,word_candidate_index_list = sentence_prune(test_file_path,0)
print("被选择的单词",word_candidate_list)
print("被选择的单词索引",word_candidate_index_list)