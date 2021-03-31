# BertPooled_Demo. 
This mini project *BertPooled_Demo* aims to reduce the computing burden of language models by simply adding a pooling layer.    
  
Though it would have been best if I pretrained a Language Model with Pooling Layers from scratch in order to proove whether such attempts are meaningful, due to realistic problems I had to settle by confirming that additional researchs may be worthwile after conducting few experiments with the concept.  
  
## Explanation.
*BertPooled_Demo* compares two models on a sentiment analysis task. The dataset for the task has been acquired from the following. http://ai.stanford.edu/~amaas/data/sentiment/. Using the report_accuracy_and_time() function in network.py would allow you to compare the required time and accuracy for each of the models.   
Both of the models uses 'bert-base-uncased' from the huggingface library to retrieve word embeddings. In the 'pool' mode a 2*2 pooling layer is inserted to halve the amount of parameters. 
