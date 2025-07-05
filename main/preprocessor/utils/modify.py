import re
import string
from tqdm import tqdm
from datasets import Dataset
def remove_punc_v2(text):
    for punc in string.punctuation:
        text = text.replace(punc, ' ')
        text = text.replace('  ', ' ')
    return text
def dupplicated_char_remover(text):
    # Thay thế các từ lặp lại và nối với ký tự a-z đứng riêng lẻ sau đó bằng dấu _
    return re.sub(r'\b(\w+)_\1\b\s([a-z])\b', r'\1_\2', text)

def preprocess_pyvi(text):
    # Bao bọc các cụm từ có dạng ký_tự/ký_tự/... bằng {}
    text = re.sub(r'(\S+/\S+(/\S+)*)', r'{\1}', text)
    return text

def postprocess_pyvi(text):
    # Khôi phục lại các cụm từ được bao bọc bởi {}
    text = re.sub(r'\{\s*(\S+(?:\s*/\s*\S+)*)\s*\}', lambda m: m.group(1).replace(' ', ''), text)
    return text

def prepare_dataset(queries, corpus, relevant_docs):
    anchors = []
    positives = []
    count = 0
    for query_id, docs in tqdm(relevant_docs.items(), desc='Processing queries'):
        for doc_id in docs:
            try:
                anchor = queries[str(query_id)]
                positive = corpus[str(doc_id)]


                anchors.append(anchor)
                positives.append(positive)

            except KeyError as e:
                count+=1
                print(f"Lỗi KeyError: {e} - Bỏ qua query_id: {query_id}, doc_id: {doc_id}")
                continue
    print(f"Total KeyError encountered: {count}")
    anchors = [ ' '.join(map(str, x)) if isinstance(x, list) else str(x) for x in anchors ]
    positives = [ ' '.join(map(str, x)) if isinstance(x, list) else str(x) for x in positives ]
    df = {
        "anchor": anchors,
        "positive": positives
    }
    
    return Dataset.from_dict(df)