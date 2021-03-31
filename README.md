# BertPooled_Demo. 
This mini project *BertPooled_Demo* aims to reduce the computing burden of language models by adding a pooling layer.    
  
Though it would have been best if I pretrained a Language Model with Pooling Layers from scratch in order to proove whether such attempts are meaningful, due to realistic problems I had to settle by confirming that additional researchs may be worthwile after conducting a small experiment with the concept.  
  
## Explanation.
*BertPooled_Demo* compares two models on a sentiment analysis task. The dataset for the task has been acquired from the following. http://ai.stanford.edu/~amaas/data/sentiment/. Using the **report_accuracy_and_time() function** in network.py would allow you to compare the required time and accuracy for each of the models. I had to conduct my experiments on colab environment and because of the limitations of RAM provided I only used 3/4 of the total train data.  

Both of the models uses 'bert-base-uncased' from the huggingface library to retrieve word embeddings.  
The 'pool' mode a 2\*2 pooling layer is inserted to halve the amount of parameters. After such processings the embeddings in the shape of (batch_size,max_length,768) is fed to a shallow neural network with a single lstm layer for the sentiment analysis task.  

## Results.  
As shown below the 'lstm_basic' model *(the one without the pooling layer)* performs slightly better than the 'pool' model *(the one with the pooling layer)* while the 'pool' model calculates faster. As experimented, it seems that the pooling layer does not result in much performance difference for simple tasks performed in small models.  
For the moment, it seems impossible to experiment with a larger model in a more complex task due to the limitations of my computer. However if possible, I am willing to proceed my project for further verfications of the idea.
![lstm_basic_result >](https://github.com/ampehta/BertPooled/blob/main/images/basic_lstm_v1.png)
![pool_result <](https://github.com/ampehta/BertPooled/blob/main/images/pool_v1.png)
