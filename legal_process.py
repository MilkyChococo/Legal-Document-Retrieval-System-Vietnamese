import pandas as pd
import os
import pickle
from tqdm import tqdm
import time
from multiprocessing import Pool, cpu_count
from main.preprocessor.preprocess import TextProcessor
import time
tqdm.pandas()
import json

train_file = r"C:\Users\GIGABYTE\OneDrive\Desktop\legal_document_retrieval - eval\data\data\train.csv"
corpus_file = r"C:\Users\GIGABYTE\OneDrive\Desktop\legal_document_retrieval - eval\data\data\corpus.csv"
test_file = r"C:\Users\GIGABYTE\OneDrive\Desktop\legal_document_retrieval - eval\data\data\test.csv"
preprocessor=TextProcessor()
def process_item(item):
    key, value = item
    value=preprocessor.preprocess_text(value)
    return key, preprocessor.post_preprocess_text(value)
def load_full_corpus(index_file):
    with open(index_file, 'rb') as f:
         batch_index = pickle.load(f)
    
    full_corpus = {}
    for batch_file in tqdm(batch_index['batch_files'], desc="Loading batches"):
         with open(batch_file, 'rb') as f:
             batch_data = pickle.load(f)
             full_corpus.update(batch_data)
    
    return full_corpus
def process_batch(batch_index_and_items):
    batch_index, batch_items = batch_index_and_items
    results = {}
    for key, value in batch_items:
        value=preprocessor.preprocess_text(value)
        results[key] = preprocessor.post_preprocess_text(value)
    return batch_index, results

def chunks(data, size):
    data_items = list(data.items())
    for i in range(0, len(data_items), size):
        yield i // size, data_items[i:i + size]

def save_batch(batch_index, batch_result, output_dir):
    batch_file = os.path.join(output_dir, f'query_batch_{batch_index}.pkl')
    with open(batch_file, 'wb') as f:
        pickle.dump(batch_result, f, protocol=pickle.HIGHEST_PROTOCOL)
    return batch_file

def load_full_corpus(index_file):
    with open(index_file, 'rb') as f:
        batch_index = pickle.load(f)

    full_corpus = {}
    for batch_file in tqdm(batch_index['batch_files'], desc="Loading batches"):
        with open(batch_file, 'rb') as f:
            batch_data = pickle.load(f)
            full_corpus.update(batch_data)

    return full_corpus
if __name__ == '__main__':
    train=pd.read_csv(train_file)
    corpora=pd.read_csv(corpus_file,usecols=['cid', 'text'])

    train = train.head(50000)
    train['cid'] = train['cid'].progress_apply(lambda x: str(x).strip("[]").strip())
    train['cid'] = train['cid'].progress_apply(lambda x: ", ".join(x.split()))
    train['text'] = train['context'].progress_apply(lambda x: str(x).strip("[]").strip("'").strip('"'))
    train['text'] = train['text'].progress_apply(lambda x: str(x).strip('"'))
    queries = {str(qid): question for qid, question in zip(train['qid'], train['question'])}
    print('Queries loaded:', len(queries))
    selected_query_ids = list(queries.keys())[:50000]
    corpora['cid'] = corpora['cid'].astype(str).str.strip()

    cids_in_train = set(train['cid'].unique())
    corpora = corpora[corpora['cid'].isin(cids_in_train)]
    corpora = {str(cid): text for cid, text in zip(corpora['cid'], corpora['text'])}
    output_dir = 'query_batches'
    os.makedirs(output_dir, exist_ok=True)

    BATCH_SIZE = 2048
    n_cores = cpu_count()

    batches = list(chunks(queries, BATCH_SIZE))

    batch_index = {
        'created_time': time.time(),
        'total_batches': len(batches),
        'batch_files': []
    }
    start=time.time()
    with Pool(processes=n_cores) as pool:
        for batch_idx, batch_result in tqdm(
            pool.imap(process_batch, batches),
            total=len(batches),
            desc="Processing and saving batches"):
            batch_file = save_batch(batch_idx, batch_result, output_dir)
            batch_index['batch_files'].append(batch_file)
            index_file = os.path.join(output_dir, 'batch_index.pkl')
            with open(index_file, 'wb') as f:
                pickle.dump(batch_index, f, protocol=pickle.HIGHEST_PROTOCOL)
        print('Done in: ',time.time()-start)
    index_file = os.path.join('query_batches', 'batch_index.pkl')

    if os.path.exists(index_file):
        full_query = load_full_corpus(index_file)
        print(len(full_query))
        print(full_query)