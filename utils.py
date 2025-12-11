import os
import re
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

def read_txt_files(dir_path, encoding='utf-8'):
    """
    Function to read texts from a directory of txt files into a pd.DataFrame
    containing the file name and the text.
    
    :param dir_path: Path to the directory containing the .txt files.
    :param encoding: Text encoding, utf-8 as standard for German text.
    """
    files = os.listdir(dir_path)

    txt_files = []
    for file in files:
        if file.endswith('.txt') and os.path.isfile(os.path.join(dir_path, file)):
            txt_files.append(file)

    texts = {}
    for filename in txt_files:
        file_path = os.path.join(dir_path, filename)
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                text = f.read()
                texts[filename] = str(text)
        except Exception as e:
            print(f"Error reading {filename}: {e}")
            continue

    texts_df = pd.DataFrame(list(texts.items()), columns=['filename', 'text'])
    
    return texts_df


def add_features(data):
    """
    Function to add features to the data for classification. These are
    keyword counts and (optional) year, if year is specified in data. Else,
    fill 'year' column with NaN values.
    """

    # add keyword counts to data
    keywords = ['klima', 'erwärmung', 'treibhaus', 'co2', 'kohle', 
                    'energiewende', 'verkehrswende', 'fridays for future',
                    'extinction rebellion']

    for key in keywords:
        count_col = key + '_count'
        data[count_col] = data['text'].apply(lambda x: len(re.findall(key, str(x).lower())))

    # add year to data
    if 'year' in data.columns:
        None
    else:
        data['year'] = np.nan

    return data

def create_embeddings(texts, embedding_file_name='embeddings.npy'):
    """
    This function creates text embeddings of the input texts using German-BERT-cased.
    Embeddings are saved to the specified filename. 
    
    :param data: Texts to embed in the form of a list of strings.
    :param embedding_file_name: File name or path to save the embeddings to. Standard is `embeddings.npy`.
    """

    # load german bert for embeddings
    tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-german-cased")
    model = AutoModel.from_pretrained("dbmdz/bert-base-german-cased")

    chunk_size = 50
    chunks = [texts[i:i+chunk_size] for i in range(0, len(texts), chunk_size)]

    all_embeddings = []

    with tqdm(total=len(chunks), desc="Embeddings chunks") as pbar:    
        
        for chunk in chunks:
            chunk_embeds = []

            for text in chunk:

                # embed texts with german-bert-cased
                inputs = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    embeddings = outputs.last_hidden_state[:, 0, :]

                chunk_embeds.append(embeddings.numpy()[0, :]) # save only first row to keep output 2d

            all_embeddings.extend(chunk_embeds)
            pbar.update(1)

    embeddings = np.array(all_embeddings)
    np.save(embedding_file_name, embeddings)   # save embeddings

    return None

def create_x(data, embeddings_file_path='embeddings.npy'):
    """
    This function creates the input matrix for the classifier by combining the 
    embeddings with their associated features. Embeddings are loaded from the specified file.
    """

    embeddings = np.load(embeddings_file_path)
    embeddings_df = pd.DataFrame(embeddings)

    # grab features from data
    features = ['year', 'klima_count','erwärmung_count',
                    'treibhaus_count','co2_count','kohle_count',
                    'energiewende_count','verkehrswende_count',
                    'fridays for future_count','extinction rebellion_count']

    data_features = data[features]

    # reset indices for concatenation
    embeddings_df.reset_index(drop=True, inplace=True)
    data_features.reset_index(drop=True, inplace=True)

    # ensure the number of rows matches
    if embeddings_df.shape[0] != data_features.shape[0]:
        raise ValueError("Embeddings and data_features do not have the same number of rows.")

    X = pd.concat([embeddings_df, data_features], axis=1)

    return X