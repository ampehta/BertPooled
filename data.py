def DataGenerator(X,y,model,tokenizer,batch_size):
    assert len(X) == len(y)
    total = math.ceil(len(X)/batch_size)
    for n in range(total):
        text_batch = X[batch_size*n:batch_size*(n+1)]
        label_batch = y[batch_size*n:batch_size*(n+1)]
        X_ = model(tf.constant(text_batch))['last_hidden_state']
        yield X_ , label_batch
