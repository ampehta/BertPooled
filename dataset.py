
import math
import tensorflow as tf

def PoolBertDataset(X, y, model, tokenizer, config, mode = 'pool'): # mode = (pool,basic.cls,basic.lstm
    assert len(X) == len(y)
    iteration = math.ceil(len(X) / config.per_iter)
    steps_per_epoch = math.ceil(len(X) / config.batch_size)

    emb = model.get_input_embeddings()
    maxpool = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), padding='valid')

    embedded_X = []
    for n in range(iteration):
        X_ = X[n * config.per_iter:(n + 1) * config.per_iter]
        X_ = tokenizer.batch_encode_plus(X_, truncation=True, padding=True, max_length=config.max_len,
                                         return_attention_mask=False, return_token_type_ids=False)['input_ids']
        if mode == 'pool': # dataset that pools the input embeddings of a bert
            X_ = [tf.squeeze(maxpool(tf.reshape(sent, (1, -1, 768, 1)))) for sent in emb(X_)]
        elif mode =='basic.cls': # dataset that retrieves the cls token from a bert
            X_ = [model(tf.reshape(tf.constant(sent), (-1, 1)))['pooler_output'] for sent in X_]
        elif mode == 'basic_lstm': # dataset that retrieves input embeddings from a bert
            X_ = emb(X_)
        embedded_X.extend(X_)

    return tf.data.Dataset.from_tensor_slices((embedded_X, y)), steps_per_epoch


def data_loader(dataset,config,train=True):
    if train:
        dataset = dataset.shuffle(100).batch(config.batch_size).repeat(config.epochs)
    else:
        dataset = dataset.batch(config.batch_size)
    for batch in dataset:
        x , y = batch
        yield x, y