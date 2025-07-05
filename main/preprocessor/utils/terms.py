import re
from tqdm import tqdm

def process_match(match, term):
    if term in ['cấp', 'hạng', 'Chương']:
        roman_numbers = re.findall(r'[IVXLCDM]+', match.group(1))
        valid_romans = filter(lambda x: re.fullmatch(r'[MDCLXVI]+', x), roman_numbers)
        return ', '.join([f"{term}_{num}" for num in valid_romans])
    
    elif term in ['Điều', 'Khoản']:
        numbers = re.findall(r'\d+', match.group(0))
        return ', '.join([f"{term}_{num}" for num in numbers])
    
    elif term == 'điểm':
        letters = re.findall(r'[a-zđ]', match.group(1)) + ([match.group(2)] if match.group(2) else [])
        return ' và '.join([f"điểm_{letter}" for letter in letters])

def terms_of_law(text):
    terms = [
        (r'[Cc]ấp\s+([IVXLCDM]+(?:,\s*[IVXLCDM]+)*(?:\s+và\s+[IVXLCDM]+)?)', 'cấp'),
        (r'[Hh]ạng\s+([IVXLCDM]+(?:,\s*[IVXLCDM]+)*(?:\s+và\s+[IVXLCDM]+)?)', 'hạng'),
        (r'[Cc]hương\s+([IVXLCDM]+(?:,\s*[IVXLCDM]+)*(?:\s+và\s+[IVXLCDM]+)?)', 'Chương'),
        (r'[Đđ]iều\s+\d+(?:,\s*\d+)*(?:\s+và\s+\d+)?', 'Điều'),
        (r'[Kk]hoản\s+\d+(?:,\s*\d+)*(?:\s+và\s+\d+)?', 'Khoản'),
        (r'[Đđ]iểm\s+([a-zđ](?:,\s*[a-zđ])*)(?:\s+và\s+([a-zđ]))?', 'điểm')
    ]

    for pattern, term in terms:
        matches = re.finditer(pattern, text)
        last_end = 0
        expanded_text_parts = []

        for match in matches:
            start, end = match.span()
            expanded_text_parts.append(text[last_end:start])
            expanded_text_parts.append(process_match(match, term))
            last_end = end
        
        expanded_text_parts.append(text[last_end:])
        text = ''.join(expanded_text_parts)

    return text
