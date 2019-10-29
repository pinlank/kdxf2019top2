
# coding: utf-8

# In[4]:


import os
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from numba import jit
import sklearn
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.metrics import mean_absolute_error

import warnings
warnings.filterwarnings('ignore')


# In[10]:


print('代码对应版本: lgb version:  2.2.3 pandas version:  0.25.0 numpy version:  1.16.2 sklearn version:  0.21.3')
print('   本机版本: ','lgb version: ', lgb.__version__, 'pandas version: ', pd.__version__,'numpy version: ', np.__version__, 'sklearn version: ', sklearn.__version__)


# In[2]:


def compute_loss1(y_hat, data):
    data = data.get_label()
    temp = np.log(np.abs(y_hat + 1)) - np.log(np.abs(data + 1))
    res = np.sqrt(np.dot(temp, temp) / len(temp))
    return 'res', res, False
# def compute_loss1(y_hat, data):
#     data = data.get_label()
#     res = mean_squared_log_error(y_hat,data)
#     return 'res', res, False
# mean_squared_log_error


# In[3]:


plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']
import matplotlib
matplotlib.rcParams['axes.unicode_minus']=False


# In[6]:


# 获取数据文件地址
def getfilelist(dir, filelist):
    newdir = dir
    if os.path.isfile(dir):
        filelist.append(dir)
    elif os.path.isdir(dir):
        for s in os.listdir(dir):
            newdir = os.path.join(dir, s)
            getfilelist(newdir, filelist)
    return filelist

#处理统计数据
def stat(data,c,name):
    data = data.sort_values()[5:-5]
    data = data - data.mean()
    if name in ['部件工作时长', '累积量参数1', '累积量参数2', '部件工作时长_diff']:
        c[name + '_max'] = data.max()
        c[name + '_min'] = data.min()
        c[name + '_mean'] = data.mean()
        c[name + '_ptp'] = data.ptp()
        c[name + '_std'] = data.std()
        c[name + '_median'] = data.median()
        c[name + '_skew'] = data.skew()#可隐藏
        c[name + '_kurt'] = data.kurt()#写成kurt
        c[name + '_absmean'] = data.abs().mean()
    else:
        c[name + '_max'] = data.max()
        c[name + '_min'] = data.min()
        c[name + '_mean'] = data.mean()
        c[name + '_ptp'] = data.ptp()
        c[name + '_std'] = data.std()
        c[name + '_median'] = data.median()
        c[name + '_skew'] = data.skew()
        c[name + '_kurt'] = data.kurt()#写成kurt
        c[name + '_absmean'] = data.abs().mean()
    if name in ['压力信号1', '压力信号2']:#
        c[name + '_05'] = data.quantile(.05)
        c[name + '_25'] = data.quantile(.25)
        c[name + '_75'] = data.quantile(.75)
        c[name + '_95'] = data.quantile(.95)
    return c
#处理单个训练样本
def process_sample_single(e,train_p):
    data = pd.read_csv(e)
    data['部件工作时长_diff'] = data['部件工作时长'].diff()
    lifemax = data['部件工作时长'].max()
    cmax=data.shape[0]
    data=data[:int(cmax*train_p)]
    data = data[data['部件工作时长'] != 0].reset_index(drop=True)
    c = {'train_file_name': os.path.basename(e)+str(train_p),
         '开关1_sum':data['开关1信号'].sum(),
         '告警1_sum':data['告警信号1'].sum(),
         '设备':data['设备类型'][0],
         'life':lifemax-data['部件工作时长'].values[-1],#max()
         }
    for i in ['部件工作时长', '累积量参数1',  '累积量参数2', '转速信号1', '转速信号2', '压力信号1',
              '压力信号2', '温度信号', '流量信号', '电流信号', 
              '部件工作时长_diff']:
            c=stat(data[i],c,i)
    this_tv_features = pd.DataFrame(c, index=[0])  
    
    return this_tv_features


# In[7]:


def get_together(listp,istest,func):

    if istest:
        rst = pd.DataFrame()
        for e in listp:
            rst = rst.append(func(e, 1))
    else:   
        train_p_list=[0.45, 0.55, 0.63, 0.68, 0.7, 0.75, 0.85]#加0.7，提0.03,修改提分较大 0.68, 0.45-0.48上分
        rst = pd.DataFrame()
        for e in listp:
            for train_p in train_p_list:
                rst = rst.append(func(e, train_p))

    return rst
def compute_loss(target, predict):
    temp = np.log(abs(target + 1)) - np.log(abs(predict + 1))
    res = np.sqrt(np.dot(temp, temp) / len(temp))
    return res


# In[8]:


start = time.time()
path = '../data/'

train_list = getfilelist(path + 'train/', [])
test_list = getfilelist(path + 'test2/', [])

func=process_sample_single
train=get_together(train_list,False,func)
test =get_together(test_list,True,func)
print("done.", time.time() - start)


# In[9]:


train['train_file_name'] = train['train_file_name'].apply(lambda x:x.split('.')[0])+'.csv'
test['train_file_name'] = test['train_file_name'].apply(lambda x:x[:-1])


# In[10]:


train = train.reset_index(drop=True)
test = test.reset_index(drop=True)
train_shape = train.shape[0]
data=train.append(test).reset_index(drop=True)
data['shebei'] = data['设备']
data=pd.get_dummies(data,columns=['设备'])


# In[14]:


train = data[:train_shape].copy()
test = data[train_shape:].copy()


# In[15]:


fe_col = [i for i in train.columns if i not in ['train_file_name', 'life', '开关2_sum', 'shebei']]
X_train = train[fe_col].copy()
y_train = np.log1p(train['life'].copy()+20)
X_test = test[fe_col].copy()
print(X_train.shape,X_test.shape)


# In[17]:


param = {
         'num_leaves': 2**7,
         'objective':'regression_l1',
         'max_depth': -1,
         'boosting': 'gbdt',
         'feature_fraction': 0.6,
         'bagging_fraction': 0.8,
         'bagging_freq':3,#3
         'metric': 'rmse',
#          'reg_sqrt':True,
#          'lambda_l1': 1,
         'lambda_l2': 10,
         'nthread': -1,
         'verbosity': -1}


# In[18]:


kf = KFold(n_splits=5, shuffle=True, random_state=1234)
# kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2019033)
oof = np.zeros(len(X_train))
predictions1 = np.zeros(len(X_test))
feature_importance_df = pd.DataFrame()
score = []
for i, (train_index, val_index) in enumerate(kf.split(X_train,y_train)):
    print("fold {}".format(i))
    X_tr, X_val = X_train.iloc[train_index], X_train.iloc[val_index]
    y_tr, y_val = y_train.iloc[train_index], y_train.iloc[val_index]
    lgb_train = lgb.Dataset(X_tr,y_tr)
    lgb_val = lgb.Dataset(X_val,y_val)
    num_round = 1000
    clf = lgb.train(param, lgb_train, num_round, valid_sets = [lgb_train, lgb_val],#feval=compute_loss1,
                    verbose_eval=100, early_stopping_rounds = 50,
                   )
    oof[val_index] = clf.predict(X_val, num_iteration=clf.best_iteration)
    score.append(clf.best_score['valid_1']['rmse'])
    fold_importance_df = pd.DataFrame()
    fold_importance_df["Feature"] = clf.feature_name()
    fold_importance_df["importance"] = clf.feature_importance()
    fold_importance_df["fold"] = i + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    
    predictions1 += clf.predict(X_test, num_iteration=clf.best_iteration) / kf.n_splits

print('train_score : ',np.mean(score))


# In[ ]:


# train_score :  0.5543862287001303 --- 最优线下结果


# In[25]:


sub = test[['train_file_name']].copy()
sub['life'] = np.expm1(predictions1).round()
sub.loc[sub['life']<200, 'life'] = ((sub.loc[sub['life']<200, 'life']/10).astype(int)*10).values
sub.to_csv('../submit/kzh5543.csv',index=False)

