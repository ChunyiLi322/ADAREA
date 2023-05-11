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
        Positive_search_space_count = 0
        data_all = []
        count_step_all = 0
        str_file = "sn"
        orignal_all_data = ""
        final_all_data = ""
        
        for data_index in range(0,16):
            orignal_all_data = orignal_all_data + data[data_index]['text1']
        
        for data_index in range(0,16): 
            
            # data_index = 1     
            args = checkpoint['args']
            interface = Interface(args)
            word_candidate_list = data[data_index]['text1'].split(' ')
        
            output_dir = '/home/lcy/adarea/simplepytorch/models/snli'
            embedding_file = os.path.join(output_dir, 'embedding.msgpack')
            word_file = os.path.join(output_dir, 'vocab.txt')
        
            for i in track(range(2), description='embedding load...'):
                with open(embedding_file, 'rb') as f:
                    embeddings = msgpack.load(f)
                time.sleep(1)
        
            for i in track(range(2), description='vocab load...'):
                vocab = Vocab.load(word_file) 
                time.sleep(1)
        
            record_data = data[data_index]['text1']
            
            
            log = open("\log_"+str_file+"_new_FSGM.txt", mode = "a+", encoding = "utf-8") 
            print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++",file = log) 
            data_k=[]
            data_k.append(data[data_index])
            # data_k.append(data[16])
            # data_k.append(data[17])
            batches = interface.pre_process(data_k, training=False)
            _, stats = model.evaluate(batches)
            print("init_stats----------------",stats, file = log)
            log.close()
            sort_word = []
            
            
            ############################首先选出梯度下降最高的点##############################################
            for word_candidate, word_candidate_index in zip(word_candidate_list, range(0,len(word_candidate_list))):
                
                word_synonym_index_list,word_synonym_list = get_word_by_candidate_name(word_candidate,embeddings, vocab , 30)
                record_data_org = data[data_index]['text1']
                word_synonym_last = word_synonym_list[29]
                record_data_1 = data[data_index]['text1'].split(' ')
                record_data_1[word_candidate_index] = word_synonym_last
                record_data_1 = " ".join(record_data_1)
                data[data_index]['text1'] = record_data_1
                data_h=[]
                data_h.append(data[data_index])
                # data_h.append(data[16])
                # data_h.append(data[17])
                batches = interface.pre_process(data_h, training=False)
                _, stats = model.evaluate(batches)
                current_loss = stats['loss']
                in_sort_word= []
                in_sort_word.append(word_candidate)
                in_sort_word.append(current_loss)
                in_sort_word.append(word_candidate_index)
                sort_word.append(in_sort_word)
                data[data_index]['text1'] = record_data_org
                

            # 根据第二元素 倒序排列
            sort_word.sort(key=lambda ele: ele[1], reverse=True)
            log = open("\log_"+str_file+"_new_FSGM.txt", mode = "a+", encoding = "utf-8")
            print("排序后的单词组别", sort_word)
            print("排序后的单词组别", sort_word, file = log)
            word_candidate_list = [i[0] for i in sort_word]
            word_candidate_index_list = [i[2] for i in sort_word]
            print("排序后的单词列表--------------",word_candidate_list, file = log)
            print("排序后的单词列表索引--------------",word_candidate_index_list, file = log)
            log.close()
            print("排序后的单词列表--------------",word_candidate_list)
            print("排序后的单词列表索引--------------",word_candidate_index_list)
        
            max_replace = 0
            sim_str_prior = 0
            record_data_need_rep = ""
        
            ############################开始替换##############################################
            for word_candidate, word_candidate_index in zip(word_candidate_list, word_candidate_index_list):                  
                log_record = open("\log_record_FSGM_"+str_file+"_baseline.txt", mode = "a+", encoding = "utf-8")
                print("---------当前句子遍历单词索引----------",word_candidate_index)
                word_synonym_index_list,word_synonym_list = get_word_by_candidate_name(word_candidate,embeddings, vocab , 30)
                print ("evaluation中的word_synonym_list的输出",word_synonym_list, file = log_record)  
                log_record.close()
                    
                for index in range(0,30):
                    
                    word_synonym_line = word_synonym_list[index]
                    record_data_1 = data[data_index]['text1'].split(' ')
                    record_data_1[word_candidate_index] = word_synonym_line
                    record_data_1 = " ".join(record_data_1)
            
                    sim_str_next = difflib.SequenceMatcher(None, record_data, record_data_1).quick_ratio()   

                               
                    log = open("\log_"+str_file+"_new_FSGM.txt", mode = "a+", encoding = "utf-8")   
                    print("这是第几句话",data_index, file = log)   
                    data[data_index]['text1'] = record_data_1   
                    print("目前对抗样本",record_data_1, file = log)  
                    log.close()
                    log = open("\log_"+str_file+"_new_FSGM.txt", mode = "a+", encoding = "utf-8")
                    data_m=[]
                    data_m.append(data[data_index])
                    # data_m.append(data[16])
                    # data_m.append(data[17])
                    print("-------------data_m-----------",data_m, file = log)
                    batches = interface.pre_process(data_m, training=False)
                    _, stats = model.evaluate(batches)
                    current_loss = stats['loss']
                    print("stats['loss']----------------",stats['loss'], file = log)
                    print("stats['score']----------------",stats['score'], file = log) 
                    log.close()
                    if stats['score'] == 1:
                       count_step_all = count_step_all + 1
                       exit = 0
                       continue
                    else:
                       data_all.append(data[data_index])
                       count_step_all = count_step_all + 1
                       exit = 1
                       print("------------跳出word训练-----------")
                       break
                
                if exit == 1:
                    print("------------跳出dataindex训练-----------")
                    break   

            print("遍历结束----------------")
            
        log = open("\log_"+str_file+"_new_FSGM.txt", mode = "a+", encoding = "utf-8") 
        batches = interface.pre_process(data, training=False)
        _, stats = model.evaluate(batches)
        current_loss = stats['loss']
        print("all_stats['loss']----------------",stats['loss'], file = log)
        print("all_stats['score']----------------",stats['score'], file = log) 
        print("count_step_all----------------",count_step_all, file = log)
        print("all_stats----------------",stats, file = log)
        print("data----------------",data, file = log)
        
        
        for data_index in range(0,16):
            final_all_data = final_all_data + data[data_index]['text1']
            
        sim_fin = difflib.SequenceMatcher(None, orignal_all_data, final_all_data).quick_ratio()
        print("orignal_all_data----------------",orignal_all_data, file = log)
        print("final_all_data----------------",final_all_data, file = log)
        print("sim_fin----------------",sim_fin, file = log)
        
        log.close()
            
