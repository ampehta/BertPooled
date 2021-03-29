
import math
import pickle
import pandas as pd
import tensorflow as tf
from data import DataGenerator
from transformers import BertTokenizer,TFBertModel

model = TFBertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train=pd.read_csv('https://raw.githubusercontent.com/ampehta/BertPooled/main/Train.csv')

train_text = train['text'].values
train_label = train['label'].values

train_encoded = tokenizer.batch_encode_plus(train_text,return_attention_mask=False,
                                            return_token_type_ids=False,truncation=True,padding=True,max_length=128)['input_ids']

batch = DataGenerator(train_encoded,train_label,model,tokenizer,32)

total = math.ceil(len(train_text)/32)
train_X = []
train_y = []
n=0
for n in range(total):
    X,y = next(batch)
    train_X.append(X)
    train_y.append(y)
    n+=1
    print(n)



with open('train_X.dat','wb') as fp:
    pickle.dump(train_X,fp)

with open('train_y.dat','wb') as fp:
    pickle.dump(train_y,fp)
