# np-lstm
- Spam classification using LSTM. 
- The LSTM Network is created using numpy. 

## Dataset
The dataset contains more than 5500 messages which are labelled as spam and ham. Follow the steps below to use the dataset: 
- Click this [link](https://www.kaggle.com/uciml/sms-spam-collection-dataset/download) to download the dataset.
- Extract the zip file. 
- Place the contents inside the extracted folder into the main repository. 

## Instructions
After preparing the dataset, train the neural network. The results will only be shown in the console. 
```
python main.py
```
If you want to do a hyperparameter search, run 'hyperparameter_optimisation.py'. The results will be logged into the log file called 'hyperparameter_search.log' 
```
python hyperparameter_optimisation.py
```
