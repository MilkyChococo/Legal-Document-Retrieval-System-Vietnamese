import pickle
import pandas as pd
with open(r'C:\Users\GIGABYTE\OneDrive\Desktop\legal_document_retrieval - eval\data\processed\corpus_after.pkl','rb') as f:
    data=pickle.load(f)
cid=[]
texts=[]
execel=[]
for x in data:
    text=data.get(x)
    execel.append([x,text])

execel=pd.DataFrame(execel,columns=['cid','text'])
#execel.to_csv('corpus_process.csv')
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import re

texts = execel['text'].astype(str)

# Làm sạch và tách từ bằng split
def clean_and_tokenize(text):
    # Loại bỏ ký tự đặc biệt và số, chuyển về chữ thường
    text = re.sub(r"[^a-zA-ZÀ-ỹà-ỹ_\s]", "", text)
    return text.lower().split()

# Gom toàn bộ từ
all_words = []
for line in texts:
    words = clean_and_tokenize(line)

    all_words.extend(words)

# Đếm tần suất
word_freq = Counter(all_words)
top_20 = word_freq.most_common(20)

# Vẽ biểu đồ
words, freqs = zip(*top_20)
plt.figure(figsize=(12,6))
plt.bar(words, freqs, color='lightblue')
plt.xticks(rotation=45)
plt.title("Frequency Chart of the 20 Most Common Words in the Dataset")
plt.xlabel("Words")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()