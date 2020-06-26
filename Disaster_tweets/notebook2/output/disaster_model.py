import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
print("Model initializing...")
max_seq_length=128
K=2

models=[0]*K
for fold in range(K):
    models[fold] = tf.keras.models.load_model('saved_model/my_model'+str(fold)+'/')

with open('saved_model/tokenizer.pickle', 'rb') as f:
    tokenizer = pickle.load(f)

def encode(texts):                
    all_tokens = []
    all_masks = []
    all_segments = []

    for text in texts:
        text = tokenizer.tokenize(text)
        text = text[:max_seq_length - 2]
        input_sequence = ['[CLS]'] + text + ['[SEP]']
        pad_len = max_seq_length - len(input_sequence)

        tokens = tokenizer.convert_tokens_to_ids(input_sequence)
        tokens += [0] * pad_len
        pad_masks = [1] * len(input_sequence) + [0] * pad_len
        segment_ids = [0] * max_seq_length

        all_tokens.append(tokens)
        all_masks.append(pad_masks)
        all_segments.append(segment_ids)

    return np.array(all_tokens), np.array(all_masks), np.array(all_segments)

def calculate(text):
    print("Analysing text...")
    data = {'text_cleaned':  [text]}
    columns=['text_cleaned']
    X = pd.DataFrame (data, columns = columns)
    X_test_encoded = encode(X['text_cleaned'].str.lower())
    y_pred = np.zeros((X_test_encoded[0].shape[0], 1))
    for model in models:
        y_pred += model.predict(X_test_encoded) / len(models)
    print("Analysis complete.")
    y_pred = np.round(y_pred).astype('int')[0,0]
    if y_pred == 1:
        print("This is a text about a disaster")
    else:
        print("This is a completely normal text")
    return y_pred

# sample data
print('Model initialized')
