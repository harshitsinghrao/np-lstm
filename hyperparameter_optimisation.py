import numpy as np
import logging

from model import Model
from handle_dataset import load_dataset
from main import train


def new_hp():
    '''
    Returns a dictionary of random, new hyperparameters.
    
    '''
    # Choose a random learning rate
    lr = 10 ** np.random.uniform(-4,-1)

    # choose a random batch size
    batch_size = np.random.choice([32, 64, 128, 256, 512])
    batch_size = batch_size.item()

    # choose a random dimension for lstm_hidden
    lstm_hidden = np.random.choice([32, 64, 128, 256, 512])
    lstm_hidden = lstm_hidden.item()
    
    # choose a random dimension for mlp_hidden
    mlp_hidden = np.random.choice([32, 64, 128, 256, 512])
    mlp_hidden = mlp_hidden.item()
    
    hp = {'lr': lr,
          'batch_size': batch_size,
          'lstm_hidden': lstm_hidden,
          'mlp_hidden': mlp_hidden}
    
    print('HYPERPARAMETERS:')
    print(f'Learning rate: {lr:.6f}\n'
          f'Batch size: {batch_size}\n'
          f'LSTM Dimension: {lstm_hidden}\n'
          f'MLP Dimension: {mlp_hidden}')
    
    logging.info('HYPERPARAMETERS:')
    logging.info(f'Learning rate: {lr:.6f}, '
                 f'Batch size: {batch_size}, '
                 f'LSTM Dimension: {lstm_hidden}, '
                 f'MLP Dimension: {mlp_hidden}')
    
    return hp    

if __name__ == '__main__':
    class Arguments:
        ''' 
        Class that will be used to hold the arguments such as hyperparameters.
        
        '''
        pass
    args = Arguments()
    
    logging.basicConfig(filename='hyperparameter_search.log', 
                        level=logging.INFO, 
                        format='%(asctime)s: %(message)s')
    
    args.decay_rate = 0.96
    args.vocab_size = 1000
    
    args.num_epochs = 20 
    args.num_hp_iters = 60 
    
    logging.info('OTHER INFORMATION')
    logging.info(f'Decay rate: {args.decay_rate}, '
                 f'Vocab size: {args.vocab_size} ')
    
    # A list that will eventually contain all the hyperparameter combinations.
    all_hp = []
    # A list that will eventually contain all the maximum accuracies achieved 
    # over all the hyperparameter combinations.  
    all_max_accs = []
    
    
    for hp_iter in range(args.num_hp_iters): 
        print('\n=====')
        print(f'HYPERPARAMETER SEARCH ITERATION: {hp_iter}\n')
        logging.info('=====')
        logging.info(f'HYPERPARAMETER SEARCH ITERATION: {hp_iter}')
        
        # Load the Dataset 
        train_set, val_set, _ = load_dataset()
        
        # Generate new hyperparameters 
        hp = new_hp()
        all_hp.append(hp)
    
        args.lr = hp['lr']
        args.batch_size = hp['batch_size']
        args.lstm_hidden = hp['lstm_hidden']
        args.mlp_hidden = hp['mlp_hidden']    
        
        # Create a new model with new hyperparameters 
        model = Model(args) 
        
        # Training loop 
        val_accs = train(model, train_set, val_set, args, hyper_search=True)
        logging.info('VALIDATION SET ACCURACIES:')
        for i in val_accs:
            logging.info(f'{i:.2f} %')
        
        max_acc = max(val_accs)
        all_max_accs.append(max_acc)
    
    
    assert len(all_max_accs) == len(all_hp) == args.num_hp_iters
    all_max_accs = np.array(all_max_accs)
    # Determine the indices of the top 5 accuracies
    best_hp_iters = (-all_max_accs).argsort()[:5]
    # Determine the best hyperparameters using those indices
    best_hp = [all_hp[i] for i in best_hp_iters] 
    # Also determine the corresponding max accuracies
    best_max_accs = all_max_accs[best_hp_iters] 
        
    # Boring Book Keeping
    print('\n=====')
    print('TOP 5 HYPERPARAMETERS, '
                 'THE CORRESPONDING MAXIMUM MEAN ACCURACIES, '
                 'AND THE CORRESPONDING HYPERPARAMETER SEARCH ITERATIONS')
    for i, j, k in zip(best_hp, best_max_accs, best_hp_iters):
        print(f'Hyperparameter combination: {i}')
        print(f'Maximum accuracy recorded: {j}')
        print(f'Corresponding hyperparameter search iteration: {k}\n')
    print('=====')
    
    logging.info('=====')
    logging.info('TOP 5 HYPERPARAMETERS, '
                 'THE CORRESPONDING MAXIMUM MEAN ACCURACIES, '
                 'AND THE CORRESPONDING HYPERPARAMETER SEARCH ITERATIONS')
    for i, j, k in zip(best_hp, best_max_accs, best_hp_iters):
        logging.info(f'Hyperparameter combination: {i}')
        logging.info(f'Maximum accuracy recorded: {j}')
        logging.info(f'Corresponding hyperparameter search iteration: {k}')
        logging.info(' ')
    logging.info('=====')