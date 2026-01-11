"""# ranking utils functions"""

# IMPORT LIBRARIES
import sys
import os
import numpy as np
import pandas as pd
import math
from scipy.optimize import curve_fit
import random
import glob
import subprocess
import matplotlib.pyplot as plt
from scipy.integrate import simps
from scipy.stats import norm
import os

import scipy

import seaborn as sns




def load_topic_target_location(topic_id,target_recall):
      ## load data

      vector_size = 100 # vector size to feed NN

      all_vectors = [[-1]*vector_size for i in range(vector_size)]

      target_location = -1 # initial


      n_docs = len(doc_rank_dic[topic_id])  # total n. docs in topic
      rel_list = rank_rel_dic[topic_id]  # list binary rel of ranked docs

      # get batches
      windows = make_windows(vector_size, n_docs)

      window_size = windows[0][1]

      # calculate batches
      rel_cnt,rel_rate, n_docs_wins = get_rel_cnt_rate(windows, window_size, rel_list)


      n_rel = sum(rel_cnt)
      prev = sum(rel_cnt)/n_docs


      #update all vector with all possible examined states
      for i in range(vector_size):
        all_vectors[i][0:i+1] = rel_rate[0:i+1] # update examined part

        #calculate target recall stopping pos
        #mark only 1st recall achieved stopping position
        if (sum(rel_cnt[0:i+1]) / sum(rel_cnt)) >= target_recall and target_location == -1:
          target_location = i


      return topic_id, n_docs, n_rel, prev, target_location

def get_rel_cnt_rate(windows, window_size, rel_list):

    # x-values are the cnt at which relevant documents occur in the window
    x = [np.sum(rel_list[w_s:w_e]) for (w_s,w_e) in windows]

    # y-values are the rate at which relevant documents occur in the window
    y = [np.sum(rel_list[w_s:w_e]) for (w_s,w_e) in windows]
    y = [y_i/window_size for y_i in y]


    # z-values are the cnt of documents in the window
    z = [len(rel_list[w_s:w_e]) for (w_s,w_e) in windows]


    # convert lists to numpy arrays
    x = np.array(x)
    y = np.array(y)
    z= np.array(z)
    return (x,y,z)




# LOAD TOPIC RELEVANCE DATA
def load_rel_data(qrels):
  qrel_fname =  os.path.join(DIR, qrels)
  with open(qrel_fname, 'r') as infile:
      qrels_data = infile.readlines()
  query_rel_dic = make_rel_dic(qrels_data) # make dictionary of list of docids relevant to each queryid

  return qrel_fname, query_rel_dic

# LOAD RUN DATA
def load_run_data(run):
  run_fname = os.path.join(DIR, run)
  with open(run_fname, 'r') as infile:
    run_data = infile.readlines()
  doc_rank_dic = make_rank_dic(run_data)  # make dictionary of ranked docids for each queryid
  rank_rel_dic = make_rank_rel_dic(query_rel_dic,doc_rank_dic) # make dic of list relevances of ranked docs for each queryid

  return doc_rank_dic, rank_rel_dic