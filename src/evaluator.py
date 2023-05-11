# coding=utf-8
# Copyright (C) 2019 Alibaba Group Holding Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from rich.progress import track
import time
import numpy as np
from src.utils.vocab import Vocab, Indexer
from src.modules.embedding import Embedding
import msgpack

from stanfordcorenlp import StanfordCoreNLP
import random
from collections import Counter

import os
from pprint import pprint
from .model import Model
from .interface import Interface
from .utils.loader import load_data
from .adduntil import get_word_by_candidate_name,get_an_optimization_adsample,word_prune,sentence_prune, get_an_initializate_sample

import difflib
import random
import math

class Evaluator:
    def __init__(self, model_path, data_file):
        self.model_path = model_path
        self.data_file = data_file

    def evaluate(self):
        data = load_data(*os.path.split(self.data_file))
        model, checkpoint = Model.load(self.model_path)
        print("evaluate中的data",data)
        # print("data中的text1",data[data_index]['text1'])
        # print("data中text1的分词",data[0]['text1'].split(' ')) 
        Positive_search_space_count = 0
        Positive_search_space_all_count = 0
        Negtive_search_space_count = 0
        Negtive_search_space_all_count = 0
        log_file_name = "snli"
        
        
        for data_index in range(0,16): 
            log_record = open("\log_record_"+ log_file_name +"_new.txt", mode = "a+", encoding = "utf-8")
            # data_index = 1     
            args = checkpoint['args']
            interface = Interface(args)
            
            output_dir = '/home/lcy/adarea/simplepytorch/models/'+ log_file_name
            embedding_file = os.path.join(output_dir, 'embedding.msgpack')
            word_file = os.path.join(output_dir, 'vocab.txt')
        
            for i in track(range(2), description='embedding load...'):
                with open(embedding_file, 'rb') as f:
                    embeddings = msgpack.load(f)
                time.sleep(1)
        
            for i in track(range(2), description='vocab load...'):
                vocab = Vocab.load(word_file) 
                time.sleep(1)
            
            
            #用于添加的平衡样本数据
            # data_n=[]
            # data_n.append(data[data_index])
            # data_n.append(data[1])
            # # data_n.append(data[2])
            # batches = interface.pre_process(data_n, training=False)
            # _, stats = model.evaluate(batches)            
            # print("用于测试的的样本ACC------------------------------------",stats)            
            
            ############################################在此循环#################################################################
            ##choose word need to change in sentence 
            word_candidate_list,word_candidate_index_list = sentence_prune(self.data_file, data_index)
                        
            # dict_word_dandidate = sentence_prune(self.data_file, 0)
            # word_candidate_list = dict_word_dandidate[1]
            # word_candidate_index_list = dict_word_dandidate[0]
           
            print ("evaluation中的word_candidate_index_list的输出",word_candidate_index_list, file = log_record)                
            print ("evaluation中的word_candidate_list的输出",word_candidate_list, file = log_record)
            Str_org = data[data_index]['text1']
            # print("test-----------------------------------------------------",data[0]['text1'][2])
           
           
            ############################################初始化对抗样本#################################################################  
            #初始化对抗样本------------------------
            data = get_an_initializate_sample(data,word_candidate_list, word_candidate_index_list,embeddings, vocab, data_index)
            
            
            data_q = []
            # data_q.append(data[16])
            # data_q.append(data[17])
            data_q.append(data[data_index])
             
            If stats['score'] == 1.0 :
               print ("该样本被舍弃----------------")
            
            
            ############################################开始遍历#################################################################
            
            
            #data[data_index]['text1'] = "a _ isuzu _ being _ across a miles"
            prior_data_loss = 0
            exit_flag = False
            top_k = 30
            negetive_data = data
            record_intial_ad = data[data_index]['text1']
            print("--------------循环开始前打印的结果",data[data_index]['text1'], file = log_record)
            search_space_all_space = len(word_candidate_list)*top_k
            current_loss = 0
            
            #正向 
            for word_candidate, word_candidate_index in zip(word_candidate_list, word_candidate_index_list):                
                #模拟退火rank(k)
                if random.randint(1,10)<3:
                    Positive_search_space_all_count = Positive_search_space_all_count + 1;
                    print("evaluate中的跳出外部循环",file = log_record)   
                    continue
                
                
                word_synonym_index_list,word_synonym_list = get_word_by_candidate_name(word_candidate,embeddings, vocab , top_k)
                print ("evaluation中的word_synonym_list的输出",word_synonym_list, file = log_record)
                
                word_synonym_list = word_prune(word_synonym_list, word_candidate)
                
                #模拟退火rank(i) [::-1] choose  most  unsimilar words
                for word_synonym_line in word_synonym_list[0:random.randint(1,30)][::-1]:
                    print("替换之前的语句",data[data_index]['text1'], file = log_record)
                    
                    prior_data = data[data_index]['text1']
                    
                    #模拟退火rank(k)
                    if random.randint(1,10)<8:
                        Positive_search_space_all_count = Positive_search_space_all_count + 1;
                        print("evaluate中的跳出索引内部循环", file = log_record)   
                        continue
                    
                    
                    
                    data = get_an_optimization_adsample(data, word_candidate_index, word_synonym_line, data_index)
                    print("替换之后的语句",data[data_index]['text1'], file = log_record)
                    
                    
                    data_n=[]
                    data_n.append(data[data_index])
                    # data_n.append(data[16])
                    # data_n.append(data[17])
                    batches = interface.pre_process(data_n, training=False)
                    _, stats = model.evaluate(batches)
                    
                    Positive_search_space_count = Positive_search_space_count + 1
                    Positive_search_space_all_count = Positive_search_space_all_count + 1;
                    
                    print("evaluate中的stats",stats, file = log_record)
                    current_loss = stats['loss']
                    
                    if stats['score'] == 1.0:
                        exit_flag = True
                        data[data_index]['text1'] = prior_data
                        continue
                    
                    sim_str_prior = difflib.SequenceMatcher(None, Str_org, prior_data).quick_ratio()
                    sim_str_next = difflib.SequenceMatcher(None, Str_org, data[data_index]['text1']).quick_ratio()
                    
                    #温度不降反升概率选择 防止局部最优
                    if  current_loss > prior_data_loss and sim_str_prior > sim_str_next and random.randint(0,int(math.exp(-(sim_str_prior - sim_str_next)/sim_str_prior))) == 0:
                        Positive_search_space_all_count = Positive_search_space_all_count + 1;
                        continue
                    pprint(stats)
                    
                    prior_data_loss = current_loss
                
                print("正向搜索空间的步数为",Positive_search_space_count, file = log_record)    
                    
                if exit_flag == True:
                    batches = interface.pre_process(data, training=False)
                    _, stats = model.evaluate(batches)
                    print("正向搜索完成的state状况--------------",stats, file = log_record) 
                    break
                
                    #反向 [::-1] choose sentence last candidate_word
            
            record_data = data[data_index]['text1']
            
            current_loss = 0
            
            for word_candidate, word_candidate_index in zip(word_candidate_list[::-1], word_candidate_index_list[::-1]):                      
                #模拟退火rank(k)
                if random.randint(1,10)<3:
                    Negtive_search_space_all_count = Negtive_search_space_all_count + 1;
                    print("evaluate中的跳出外部循环", file = log_record)   
                    continue    
                word_synonym_index_list,word_synonym_list = get_word_by_candidate_name(word_candidate,embeddings, vocab , top_k)
                print ("evaluation中的word_synonym_list的输出",word_synonym_list, file = log_record)
                word_synonym_list = word_prune(word_synonym_list, word_candidate)
                
                #模拟退火rank(i) [::-1] choose  most  unsimilar words
                for word_synonym_line in word_synonym_list[0:random.randint(1,30)][::-1]:
                    prior_data = negetive_data[data_index]['text1']
                    print("替换之前的语句",negetive_data[data_index]['text1'], file = log_record)
                    
                    #模拟退火rank(k)
                    if random.randint(1,10)<8:
                        Negtive_search_space_all_count = Negtive_search_space_all_count + 1;
                        print("evaluate中的跳出索引内部循环", file = log_record)   
                        continue
                    
                    negetive_data = get_an_optimization_adsample(negetive_data, word_candidate_index, word_synonym_line, data_index)
                    print("替换之后的语句",negetive_data[data_index]['text1'], file = log_record)
                    
                    data_m=[]
                    # data_m.append(data[16])
                    # data_m.append(data[17])
                    data_m.append(negetive_data[data_index])
                    
                    batches = interface.pre_process(data_m, training=False)
                    _, stats = model.evaluate(batches)
                    
                    
                    Negtive_search_space_count = Negtive_search_space_count + 1
                    Negtive_search_space_all_count = Negtive_search_space_all_count + 1;
                    print("evaluate中的stats",stats, file = log_record)
                    current_loss = stats['loss']
                       
                    if stats['score'] == 1.0:
                        exit_flag = True
                        negetive_data[data_index]['text1'] = prior_data
                        continue
                    
                    sim_str_prior = difflib.SequenceMatcher(None, Str_org, prior_data).quick_ratio()
                    sim_str_next = difflib.SequenceMatcher(None, Str_org, negetive_data[data_index]['text1']).quick_ratio()
                    
                    #温度不降反升概率选择 防止局部最优
                    if  current_loss > prior_data_loss and sim_str_prior > sim_str_next and random.randint(0,int(math.exp(-(sim_str_prior - sim_str_next)/sim_str_prior))) == 0:
                        Negtive_search_space_all_count = Negtive_search_space_all_count + 1;
                        continue
                    pprint(stats)
                    
                    
                prior_data_loss = current_loss
                print("反向搜索空间的步数为",Negtive_search_space_count, file = log_record)    
                    
                if exit_flag == True:
                    batches = interface.pre_process(negetive_data, training=False)
                    _, stats = model.evaluate(batches)
                    print("反向搜索完成的state状况--------------",stats, file = log_record) 
                    break    
                
            log_record.close()    
            log = open("\log_"+ log_file_name +"_new.txt", mode = "a+", encoding = "utf-8")   
            print("这是第几句话",data_index, file = log)  
            print("整个空间（所有样本，包括非对抗样本）",search_space_all_space, file = log) 
            print("原始样本",Str_org, file = log)
            print("初始化对抗样本",record_intial_ad, file = log)
            print("正向搜索空间的结果",record_data)
            print("正向搜索空间的结果",record_data, file = log)
            print("正向搜索空间的步数 约简+不约简",Positive_search_space_all_count, file = log)    
            print("正向搜索空间最终  不约简",Positive_search_space_count, file = log)
            Positive_search_space_all_count = 0
            Positive_search_space_count = 0
            print("反向搜索空间的结果",negetive_data[data_index]['text1'])
            print("反向搜索空间的结果",negetive_data[data_index]['text1'], file = log)
            print("反向搜索空间的步数最终为 约简+不约简",Negtive_search_space_all_count, file = log)    
            print("反向搜索空间最终为   不约简",Negtive_search_space_count, file = log)
            data_q = []
            # data_q.append(data[16])
            # data_q.append(data[17])
            data_q.append(data[data_index])
            batches = interface.pre_process(data_q, training=False)
            _, stats = model.evaluate(batches)
            print("准确率-------------",stats, file = log)
            Negtive_search_space_all_count = 0
            Negtive_search_space_count = 0
            print("遍历结束----------------")
            log.close()