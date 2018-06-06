
# coding: utf-8

# In[2]:


import gc
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.metrics import roc_auc_score
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


# In[3]:


gc.enable()
buro_bal = pd.read_csv('./input/bureau_balance.csv')
print('Buro bal shape : ', buro_bal.shape)


# In[4]:


print('transform to dummies')
#buro_bal = pd.concat([buro_bal, pd.get_dummies(buro_bal.STATUS, prefix='buro_bal_status')], 
#                     axis=1).drop('STATUS', axis=1)
buro_bal = pd.concat([buro_bal, pd.get_dummies(buro_bal.STATUS, prefix='buro_bal_status')], axis=1)
buro_bal = buro_bal.drop('STATUS', axis=1)
buro_bal.head()


# In[5]:


print('Counting buros')
#computing counts of MONTHS_BALANCE
buro_counts = buro_bal[['SK_ID_BUREAU', 'MONTHS_BALANCE']].groupby('SK_ID_BUREAU').count()
#following, I don't know how to compute the result with map function
buro_bal['buro_count'] = buro_bal['SK_ID_BUREAU'].map(buro_counts['MONTHS_BALANCE'])
buro_bal.head()


# In[6]:


print('averaging buro bal')
#computing average values of buro_bal's columns
avg_buro_bal = buro_bal.groupby('SK_ID_BUREAU').mean()
avg_buro_bal.head()
#notice that we have two layer index now in the following table


# In[7]:


#change to the average name 
avg_buro_bal.columns = ['avg_buro_' + f_ for f_ in avg_buro_bal.columns]
#buro_bal is finished, we have avg_buro_bal
del buro_bal
gc.collect()

avg_buro_bal.head()


# In[8]:


print('Read Bureau')
buro = pd.read_csv('./input/bureau.csv')
buro.head()


# In[9]:


print('Go to dummies')
buro_credit_active_dum = pd.get_dummies(buro.CREDIT_ACTIVE, prefix='ca_')
buro_credit_active_dum.head()


# In[10]:


buro_credit_currency_dum = pd.get_dummies(buro.CREDIT_CURRENCY, prefix='cu_')
buro_credit_currency_dum.head()


# In[11]:


buro_credit_type_dum = pd.get_dummies(buro.CREDIT_TYPE, prefix='ty_')
buro_credit_type_dum.head()


# In[12]:


buro_full = pd.concat([buro, buro_credit_active_dum, buro_credit_currency_dum, buro_credit_type_dum], axis=1)
del buro_credit_active_dum, buro_credit_currency_dum, buro_credit_type_dum
gc.collect()
buro_full.head()


# In[13]:


print('Merge with buro avg')
#suffixes is used for duplicated names of columns in two DataFrame, adding posting fixes.
buro_full = buro_full.merge(right=avg_buro_bal.reset_index(), how='left', 
                            on='SK_ID_BUREAU', suffixes=('', '_bur_bal'))
buro_full.head()


# In[14]:


print('Counting buro per SK_ID_CURR')
nb_bureau_per_curr = buro_full[['SK_ID_CURR', 'SK_ID_BUREAU']].groupby('SK_ID_CURR').count()
#how many SK_ID_BUREAU does every SK_ID_CURR have? then covered the SK_ID_BUREAU
buro_full['SK_ID_BUREAU'] = buro_full['SK_ID_CURR'].map(nb_bureau_per_curr['SK_ID_BUREAU'])
buro_full.head()


# In[15]:


print('Averaging bureau')
#mean will delete article features
avg_buro = buro_full.groupby('SK_ID_CURR').mean()
avg_buro.head()#article features is deleted


# In[16]:


del buro, buro_full
gc.collect()


# In[17]:


print('Read prev')
prev = pd.read_csv('./input/previous_application.csv')
prev.head()


# In[18]:


#find object type feature to do dummies
prev_cat_features = [
    f_ for f_ in prev.columns if prev[f_].dtype == 'object'
]
prev_cat_features


# In[19]:


print('Go to dummies')
prev_dum = pd.DataFrame()
for f_ in prev_cat_features:
    prev_dum = pd.concat([prev_dum, pd.get_dummies(prev[f_], prefix=f_).astype(np.uint8)], axis=1)
print(prev_dum.shape)
prev_dum.head()


# In[20]:


prev = pd.concat([prev, prev_dum], axis=1)
del prev_dum
gc.collect()


# In[21]:


prev.head()


# In[22]:


print('Counting number of Prevs')
nb_prev_per_curr = prev[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()
prev['SK_ID_PREV'] = prev['SK_ID_CURR'].map(nb_prev_per_curr['SK_ID_PREV'])
prev.head()


# In[23]:


print('Averaging prev')
avg_prev = prev.groupby('SK_ID_CURR').mean()
avg_prev.head()


# In[24]:


del prev
gc.collect()


# In[25]:


print('Reading POS_CASH')
pos = pd.read_csv('./input/POS_CASH_balance.csv')
pos.head()


# In[26]:


print('Go to dummies')
pos = pd.concat([pos, pd.get_dummies(pos['NAME_CONTRACT_STATUS'])], axis=1)
pos.head()


# In[27]:


print('Compute nb of prevs per curr')
nb_prevs = pos[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()
pos['SK_ID_PREV'] = pos['SK_ID_CURR'].map(nb_prevs['SK_ID_PREV'])
print('Go to averages')
avg_pos = pos.groupby('SK_ID_CURR').mean()

del pos, nb_prevs
gc.collect()
avg_pos.head()


# In[28]:


print('Reading CC balance')
cc_bal = pd.read_csv('./input/credit_card_balance.csv')
cc_bal.head()


# In[29]:


print('Go to dummies')
cc_bal = pd.concat([cc_bal, pd.get_dummies(cc_bal['NAME_CONTRACT_STATUS'], prefix='cc_bal_status_')], axis=1)

nb_prevs = cc_bal[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()
cc_bal['SK_ID_PREV'] = cc_bal['SK_ID_CURR'].map(nb_prevs['SK_ID_PREV'])

print('Compute average')
avg_cc_bal = cc_bal.groupby('SK_ID_CURR').mean()
avg_cc_bal.columns = ['cc_bal_' + f_ for f_ in avg_cc_bal.columns]

del cc_bal, nb_prevs
gc.collect()
avg_cc_bal.head()


# In[30]:


print('Reading Installments')
inst = pd.read_csv('./input/installments_payments.csv')
nb_prevs = inst[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()
inst['SK_ID_PREV'] = inst['SK_ID_CURR'].map(nb_prevs['SK_ID_PREV'])

avg_inst = inst.groupby('SK_ID_CURR').mean()
avg_inst.columns = ['inst_' + f_ for f_ in avg_inst.columns]
avg_inst.head()


# In[31]:


print('Read data and test')
train = pd.read_csv('./input/application_train.csv')
test = pd.read_csv('./input/application_test.csv')
print('Shapes : ', train.shape, test.shape)


# In[32]:


print('Read data and test')
data = pd.read_csv('./input/application_train.csv')
test = pd.read_csv('./input/application_test.csv')
print('Shapes : ', data.shape, test.shape)


# In[33]:


categorical_feats = [
    f for f in data.columns if data[f].dtype == 'object'
]#find all object feature
for f_ in categorical_feats:
    #use factorize for changing object feature to numeric feature
    #it's different from dummies, only generate one column numeric feature
    train[f_], indexer = pd.factorize(train[f_])
    test[f_] = indexer.get_indexer(test[f_])
train.head()


# In[34]:


#merge all table
train = train.merge(right=avg_buro.reset_index(), how='left', on='SK_ID_CURR')
test = test.merge(right=avg_buro.reset_index(), how='left', on='SK_ID_CURR')

train = train.merge(right=avg_prev.reset_index(), how='left', on='SK_ID_CURR')
test = test.merge(right=avg_prev.reset_index(), how='left', on='SK_ID_CURR')

train = train.merge(right=avg_pos.reset_index(), how='left', on='SK_ID_CURR')
test = test.merge(right=avg_pos.reset_index(), how='left', on='SK_ID_CURR')

train = train.merge(right=avg_cc_bal.reset_index(), how='left', on='SK_ID_CURR')
test = test.merge(right=avg_cc_bal.reset_index(), how='left', on='SK_ID_CURR')

train = train.merge(right=avg_inst.reset_index(), how='left', on='SK_ID_CURR')
test = test.merge(right=avg_inst.reset_index(), how='left', on='SK_ID_CURR')

del avg_buro, avg_prev
gc.collect()
train.head()


# In[35]:


print('train shape is ', train.shape)
print('test shape is ', test.shape)


# In[36]:


ID = test.SK_ID_CURR


# In[37]:


train.columns


# In[38]:


#change to _
train.columns = train.columns.str.replace('[^A-Za-z0-9_]', '_')
test.columns = test.columns.str.replace('[^A-Za-z0-9_]', '_')


# In[39]:


train.columns


# In[40]:


floattypes = []
inttypes = []
stringtypes = []
for c in test.columns:
    if(train[c].dtype=='object'):
        train[c] = train[c].astype('str')
        stringtypes.append(c)
    elif(train[c].dtype=='int64'):
        train[c] = train[c].astype('int32')
        inttypes.append(c)
    else:
        train[c] = train[c].astype('float32')
        floattypes.append(c)
print(floattypes)


# In[41]:


train.head()


# In[ ]:


train = train.reset_index(drop=True)
test = test.reset_index(drop=True)
train.head()


# In[ ]:


train.fillna(0)
test.fillna(0)


# In[ ]:


from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier

train = train.drop('TARGET', axis=1)
train_labels = train['TARGET']
train_labels = np.array(train_labels).reshape((-1,))
folds = KFold(n_splits=5, shuffle=True, random_state=50)
valid_preds = np.zeros(train.shape[0])
test_preds = np.zeros(test.shape[0])

for n_fold, (train_indices, valid_indices) in enumerate(folds.split(train)):
    train_fold, train_fold_labels = train[train_indices, :], train_labels[train_indices]
    valid_fold, valid_fold_labels = train[valid_indices, :], train_labels[valid_indices]
    clf = LGBMClassifier(
        n_estimators=1000,
        learning_rate = 0.005,
        num_leaves=80,
        colsample_bytree=.8,
        subsample=.9,
        max_depth=7,
        reg_alpha=.1,
        reg_lambda=.1,
        min_split_gain=.01,
        min_child_weight=2
    )
    
    clf.fit(train_fold, train_fold_labels,
            eval_set = [(train_fold, train_fold_labels), (valid_fold, valid_fold_labels)],
            eval_metric='auc', early_stopping_rounds=50, verbose=False)
    valid_preds[valid_indices] = clf.predict_proba(valid_fold, num_iteration=clf.best_iteration_)[:,1]
    test_preds += clf.predict_proba(test, num_iteration=clf.best_iteration_)[:, 1]/folds.n_splits
    print('Fold %d AUC : %0.6f' % (n_fold+1, roc_auc_score(valid_fold_labels, valid_preds[valid_indices])))
    del clf,train_fold, train_fold_labels, valid_fold, valid_fold_labels
    gc.collect()
    
submission = app_test[['SK_ID_CURR']]
submission['TARGET'] = test_preds
submission.to_csv("best_baseline.csv", index=False)

