import pandas as pd
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

with open('sim_log_sc.csv','rt') as csvfile:
    reader = csv.DictReader(csvfile) 
    column_x = [int(row['iteration']) for row in reader]

with open('sim_log_sn.csv','rt') as csvfile:
    reader = csv.DictReader(csvfile)
    column_nn = [float(row['Sim']) for row in reader]

with open('sim_log_sc.csv','rt') as csvfile:
    reader = csv.DictReader(csvfile) 
    column_org = [float(row['Sim']) for row in reader]
    
with open('sim_log_qq.csv','rt') as csvfile:
    reader = csv.DictReader(csvfile) 
    column_vv = [float(row['Sim']) for row in reader]

with open('sim_log_wk.csv','rt') as csvfile:
    reader = csv.DictReader(csvfile) 
    column_jj = [float(row['Sim']) for row in reader]

# print(column_x)
# print(column_jj)



fig = plt.figure(figsize=(10,5))
plt.xlabel(u'x-iteration',fontsize=14)
plt.ylabel(u'y-Sim',fontsize=14)

in1, = plt.plot(column_x[0:50],column_org[0:50],color="maroon",linewidth=0.9,linestyle='-')
in2, = plt.plot(column_x[0:29],column_nn[0:29],color="steelblue",linewidth=0.5,linestyle='-')
in3, = plt.plot(column_x[0:42],column_vv[0:42],color="darkgoldenrod",linewidth=0.5,linestyle='-')
in4, = plt.plot(column_x[0:29],column_jj[0:29],color="darkgreen",linewidth=0.5,linestyle='-')

# plt.legend(handles = [in1,in2,in3,in4],labels=['basic(Transformer)','basic+C_noun/0.0433','basic+C_verb/0.0245','basic+C_adj+adv/0.0212'],loc=2)
plt.legend(handles = [in1,in2,in3,in4],labels=['SNLI','SciTail','Quora','WikiQA'],loc=2)
plt.gca().spines['left'].set_color('y')
plt.gca().spines['right'].set_color('g')
plt.gca().spines['bottom'].set_color('c')
plt.gca().spines['top'].set_color('b')

plt.grid(True,linestyle = "--",color = 'gray' ,linewidth = '0.5',axis='both')

# ax1=fig.add_axes([0.5,0.45,0.35,0.25])
# ax1.plot(column_x[600:690],column_org[600:690],color="maroon",linewidth=0.9,linestyle='-',marker='*')
# ax1.plot(column_x[600:690],column_nn[600:690],color="steelblue",linewidth=0.5,linestyle='--',marker='1')
# ax1.plot(column_x[600:690],column_vv[600:690],color="darkgoldenrod",linewidth=0.5,linestyle='-.',marker='+')
# ax1.plot(column_x[600:690],column_jj[600:690],color="darkgreen",linewidth=0.5,linestyle=':',marker='<')

# ax1.set_title("partial enlargement")
# ax1.grid(True,linestyle = "--",color = 'gray' ,linewidth = '0.5',axis='both')

plt.savefig('./zhexian.pdf')
