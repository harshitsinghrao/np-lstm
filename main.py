import numpy as np

from model import Model
from handle_dataset import load_dataset
from functions import cross_entropy
from handle_dataset import shuffle


class EvaluatePerformance:
    '''
    Evaluate performance on validation or the test set. 
    
    '''
    def __init__(self):
        self.accuracy = 0
        self.correct = 0
        self.total = 0
    
    def track_accuracy(self, pred, y, args):
        # Determine how many times our prediction was correct. 
        pred = np.argmax(pred, axis=1)
        y = np.argmax(y, axis=1)
        self.correct += (y==pred).sum()
        
        # Calculate accuracy
        self.total += args.batch_size
        self.accuracy = 100 * self.correct/self.total

    def evaluate(self, model, eval_set, args):
        '''
        Run the model on the eval_set. Here eval_set is the validation or the
        test set. 

        '''
        x_eval, y_eval = eval_set
        no_of_eval_samples = x_eval.shape[0]
        
        x_eval, y_eval = shuffle(x_eval, y_eval)
        for i in range(0, no_of_eval_samples, args.batch_size):
            x, y = x_eval[i:i+args.batch_size], y_eval[i:i+args.batch_size]
            pred = model(x)
            self.track_accuracy(pred, y, args)
            
        return self.accuracy
            

def train(model, train_set, val_set, args, hyper_search=False):

    ''' 
    Training; check performance on the validation dataset after every epoch.  
    
    hyper_search: Whether to optimise the hyperparameters or not. When set
    to True, val_accs will record accuracy on the validation set after
    every epoch, which is only required in the case of hyperparameter search.
    
    '''
    x_train, y_train = train_set
    no_of_train_samples = x_train.shape[0]
    
    # val_accs is only relevant in the case of hyperparameter optimisation.
    if hyper_search: val_accs = []
    
    for epoch in range(args.num_epochs):
        model.reset_rms_cache()
        x_train, y_train = shuffle(x_train, y_train)
        
        if not hyper_search:
            print(f'\n***\nEPOCH: {epoch+1}\nTRAIN SET')
        
        for i in range(0, no_of_train_samples, args.batch_size):    
            x, y = x_train[i:i+args.batch_size], y_train[i:i+args.batch_size]
            
            pred = model(x, dropout=True)
            loss = cross_entropy(pred,y)
            model.backward(y, dropout=True)
            model.update(args.lr, args.decay_rate)
            
            if i % 10 == 0 and not hyper_search:
                print(f'Iteration: {i}, loss: {loss:.4f}')    
                    
        # Evaluate performance on the validation set after every epoch
        evaluate_performance = EvaluatePerformance()
        accuracy = evaluate_performance.evaluate(model, val_set, args)
        print(f'\nVALIDATION SET\nAccuracy: {accuracy:.2f} %')
        
        if hyper_search: val_accs.append(accuracy)    
        
    if hyper_search: return val_accs


if __name__ == '__main__':
    class Arguments:
        ''' 
        Class that will be used to hold the arguments such as hyperparameters.
        
        '''
        pass
    args = Arguments()
    
    # Hyperparameters and other values. 
    args.lr = 12e-4
    args.batch_size = 32
    args.lstm_hidden = 256
    args.mlp_hidden = 512
    
    args.decay_rate = 0.96
    args.vocab_size = 1000
    args.num_epochs = 20 
    
    # Load the Dataset
    train_set, val_set, test_set = load_dataset()
    
    # Initialise the model 
    model = Model(args) 
    
    # Training loop
    train(model, train_set, val_set, args)
    
    # After training is finished, evaluate performance on the test set. 
    print('\n***\nTEST SET')
    evaluate_performance = EvaluatePerformance()
    accuracy = evaluate_performance.evaluate(model, test_set, args)
    print(f'Accuracy: {accuracy:.2f} %')