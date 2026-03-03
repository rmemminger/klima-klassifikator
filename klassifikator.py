import argparse
import os
import pandas as pd
import xgboost as xgb

from utils import read_txt_files, add_features, create_embeddings, create_x

def setup_argparse():

    parser = argparse.ArgumentParser()
    
    input_group = parser.add_mutually_exclusive_group(required=True)
    
    # Specify if reading from .csv or txt file
    input_group.add_argument('--csv', type=str, metavar='FILEPATH')
    input_group.add_argument('--txt', type=str, metavar='DIRPATH')
    
    # Embedding options: create new embeddings or load existing ones (faster)
    embed_group = parser.add_argument_group('Embedding')
    embed_group.add_argument('--embed', action='store_true')
    embed_group.add_argument('--load-embed', type=str, metavar='FILEPATH')
    
    return parser.parse_args()

def check_args(args):
    
    if args.csv and not os.path.isfile(args.csv):
        raise FileNotFoundError(f"CSV file not found: {args.csv}")
    
    if args.txt and not os.path.isdir(args.txt):
        raise FileNotFoundError(f"Folder not found: {args.txt}")
    
    if args.load_embed and not os.path.isfile(args.load_embed):
        raise FileNotFoundError(f"Embedding file not found: {args.load_embed}")

    if not args.embed and not args.load_embed:
        print("Warning: Neither --embed nor --load-embed specified. No embedding operations will be performed.")
    
    if args.embed and args.load_embed:
        raise argparse.ArgumentError(None, "Cannot use both --embed and --load-embed. Choose one.")
    
    return args

def main():

    args = setup_argparse()
    
    try:
        args = check_args(args)
    except (FileNotFoundError, argparse.ArgumentError) as e:
        print(f"Error: {e}")
        return
    
    if args.csv:
        data = pd.read_csv(args.csv, index_col=0)
    elif args.txt:
        data = read_txt_files(dir_path=args.txt, encoding='utf-8')
    else:
        print("Input error.")
        return

    data = add_features(data)
    data.head()

    texts = data['text'].to_list()

    if args.embed:
        create_embeddings(texts)
        embed_path = 'embeddings.npy'
    elif args.load_embed:
        embed_path = args.load_embed

    X = create_x(data=data, embeddings_file_path=embed_path)

    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model('xgb_model.json')
    preds = xgb_model.predict(X)    #make predictions

    # concatenate data with predicted label and save
    preds_df = pd.DataFrame(preds, columns=['pred_y'])

    preds_df.reset_index(drop=True, inplace=True)
    data.reset_index(drop=True, inplace=True)

    result = pd.concat([data, preds_df], axis=1)

    # NEW SETTING: Safety measure: include at least one (1) keyword on top of pred_y = 1
    # for this add new classification column to easily understand difference between
    # pred_y and safety rule

    keywords = ['klima', 'erwärmung', 'treibhaus', 'co2', 'kohle', 
          'energiewende', 'verkehrswende', 'fridays for future',
          'extinction rebellion']
    keycols = [kw + '_count' for kw in keywords]
    result['class'] = 0
    # if pred_y == 1 and at least 1 keyword, class = 1, else default 0
    result.loc[(result['pred_y'] == 1) & (result[keycols].sum(axis=1) > 0), 'class'] = 1

    # create output folder if not yet existing
    if not os.path.isdir('output'):
        os.makedirs('output')

    # save results to output folder
    result.to_csv('output/texts_classified.csv')
    result[result['class']==1].to_csv('output/texts_climate.csv')
    
    return

if __name__ == "__main__":
    main()
