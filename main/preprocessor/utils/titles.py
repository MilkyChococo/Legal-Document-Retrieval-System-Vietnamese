import re
from typing import List
from underthesea import ner
from main.preprocessor.vocab.title import TITLES

def titles_terms(text):
    """
    Chuẩn hóa các từ khóa nhiệm vụ trong văn bản bằng cách thay thế
    bằng dạng nối dấu gạch dưới, không phân biệt hoa thường.
    """
    for item in TITLES:
        # Tạo regex để tìm kiếm từ khóa, không phân biệt hoa thường, chỉ khớp nguyên từ
        pattern = re.compile(r'\b' + re.escape(item) + r'\b', re.IGNORECASE)
        # Hàm thay thế giữ nguyên chữ hoa/thường ban đầu
        def repl(match):
            matched = match.group(0)
            # Nếu matched là viết hoa toàn bộ thì giữ nguyên
            if matched.isupper():
                return item.replace(" ", "_").replace("-", "").upper()
            # Nếu matched là viết thường toàn bộ thì giữ nguyên
            elif matched.islower():
                return item.replace(" ", "_").replace("-", "").lower()
            # Ngược lại, trả về dạng chuẩn hóa
            return item.replace(" ", "_").replace("-", "")
        text = pattern.sub(repl, text)
    return text

def ner_tokenize(text):
    """
    Chuẩn hóa các thực thể tên riêng trong văn bản bằng cách thay thế
    bằng dạng nối dấu gạch dưới, không phân biệt hoa thường.
    """
    ner_results = ner(text)
    entities: List[str] = []
    for item in ner_results:
        # Kiểm tra nếu thực thể là tên riêng (B-NP với Np hoặc B-PER)
        if (item[2] == 'B-NP' and item[1] == 'Np') or item[-1] == 'B-PER':
            entity = item[0]
            if entity not in entities:
                entities.append(entity)
    # Thay thế từng thực thể, ưu tiên thực thể dài hơn để tránh lồng nhau
    entities.sort(key=lambda x: -len(x))
    for entity in entities:
        pattern = re.compile(r'\b' + re.escape(entity) + r'\b', re.IGNORECASE)
        def repl(match):
            matched = match.group(0)
            if matched.isupper():
                return entity.replace(" ", "_").replace("-", "").upper()
            elif matched.islower():
                return entity.replace(" ", "_").replace("-", "").lower()
            return entity.replace(" ", "_").replace("-", "")
        text = pattern.sub(repl, text)
    return text