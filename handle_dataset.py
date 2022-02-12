import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import TreebankWordTokenizer

   
def get_x(df, vocab_size=1000, seq_len=150):
    '''
    Args: 
        vocab_size (int): Number of words in the vocabulary
        seq_len (int): Same length for each sentence
    
    Returns:
        x_train (np array): array of size (train_len, seq_len)
        x_test (np array): array of size (5572 - train_len, seq_len)
    '''
    # CREATE VOCABULARY
    alltext = df['v2'].str.cat(sep=' ')
    alltext = TreebankWordTokenizer().tokenize(alltext)
    vocab = list(nltk.FreqDist(alltext).most_common(vocab_size))
    vocab = {vocab[i][0]: i+1 for i in range(vocab_size)} 
    
    # GENERATE X
    x = df['v2'].tolist()
    concat_x = []
    for sentence in x:
        
        # String to list conversion
        sentence = TreebankWordTokenizer().tokenize(sentence)
        
        # map words to corresponding values from vocabulary
        for i, word in enumerate(sentence):
            if word in vocab.keys():
                sentence[i] = vocab[word]
            else:
                sentence[i] = 0
        
        # Ensure all the sentences are of same length ie. seq_len
        if len(sentence) < seq_len:
            sentence = np.pad(sentence, (seq_len - len(sentence), 0), 'constant')
        else:
            sentence = sentence[:seq_len]
        
        concat_x.append(sentence)
        
    concat_x = np.stack(concat_x)
    return concat_x
    
        
def get_y(df):
    ''' 
    Returns:
        y_train (np array): shape (train_len, 2)
        y_test (np array): shape (5572 - train_len, 2)
    '''
    labels = df['v1'].replace(to_replace=('ham','spam'), value=(0,1))
    labels = np.array(labels)
    
    labels_one_hot = np.zeros((labels.shape[0], 2))
    labels_one_hot[np.arange(labels.shape[0]), labels] = 1
    
    return labels_one_hot


def shuffle(x, y):
    '''
    Shuffles the rows of two arrays in unison

    Parameters
    ----------
    x : numpy array
    y : numpy array

    Returns
    -------
    Shuffled arrays

    '''
    assert x.shape[0] == y.shape[0]
    rng = np.random.default_rng()
    rng = rng.permutation(x.shape[0])
    x, y = x[rng], y[rng]
    return x, y


def split_dataset(x, y): 
    '''
    Split the dataset into training, validation, and test sets. 
    20% of the dataset will be used for the test set.
    The remaining dataset will be subdivided into training set and validation set. 
    The ratio of training and validation set will be 80:20

    Parameters
    ----------
    x : np array of shape (total samples, sequence length)
    y : labels; np array of shape (total samples, no of labels)

    Returns
    -------
    
    train_set : list [x_train, y_train]
    Description : x_train; np array of shape (no of training examples, sequence length)
                  y_train; np array of shape (no of training examples, no of labels)
    
    validation_set : list [x_validation, y_validation]
    Description : x_validation; np array of shape (no of validation examples, sequence length)
                  y_validation; np array of shape (no of validation examples, no of labels)
    
    test_set : list [x_test, y_test]
    Description : x_test; np array of shape (no of test examples, sequence length)
                  y_test; np array of shape (no of test examples, no of labels)

    '''
    assert x.shape[0] == y.shape[0]
    total_no_of_samples = x.shape[0]
    no_of_train_samples = round(total_no_of_samples * 0.8 * 0.8)
    no_of_validation_samples = round(total_no_of_samples * 0.8 * 0.2)
    
    print('DATASET SPECIFICATIONS')
    print(f'Total number of samples: {total_no_of_samples}')
    print(f'Training split: {no_of_train_samples} samples')
    print(f'Validation split: {no_of_validation_samples} samples')
    print(f'Test split: {total_no_of_samples - (no_of_train_samples + no_of_validation_samples)} samples\n')

    train_set, validation_set, test_set = [], [], []
    for arr in x, y:
        # Training split
        arr_train = arr[:no_of_train_samples]
        train_set.append(arr_train)
        
        # Validation split
        arr_validation = arr[no_of_train_samples: no_of_train_samples + no_of_validation_samples]
        validation_set.append(arr_validation)
        
        # Test split
        arr_test = arr[no_of_train_samples + no_of_validation_samples:]
        test_set.append(arr_test)

    return train_set, validation_set, test_set

def load_dataset():
    # Load the dataset
    df = pd.read_csv('spam.csv', delimiter=',', encoding='latin-1') 
    # remove the redundant columns
    df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True) 
    # Transform the dataset
    
    x, y = get_x(df), get_y(df) 
    # Shuffle the dataset
    x, y = shuffle(x, y) 
    # Split the dataset into training, validation, and test sets.
    train_set, validation_set, test_set = split_dataset(x, y)
    
    return train_set, validation_set, test_set