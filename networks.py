import time
from dataset import PoolBertDataset,data_loader
from model import make_model

def report_accuracy_and_time(train_X,train_y,test_X,test_y,bert,tokenizer,config,mode):
    start_time = time.perf_counter()
    dataset, steps_per_epoch = PoolBertDataset(train_X, train_y, bert, tokenizer, config, mode)

    dataloader = data_loader(dataset,config,train=True)
    model = make_model(config.learning_rate)
    model.fit(dataloader,epochs=config.epochs,verbose=0,steps_per_epoch=steps_per_epoch)

    dataset_test, steps_per_epoch = PoolBertDataset(test_X, test_y, bert, tokenizer, config, mode)
    dataloader_test = data_loader(dataset_test, config, train=False)
    output = model.evaluate(dataloader_test,return_dict=True)
    spent_time = (time.perf_counter() - start_time)/60
    print('##############################')
    print(f"Time Spent: {spent_time} minutes\nAccuracy :{output['accuracy']}\nTotal Parameters :{model.count_params()}")
