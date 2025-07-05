from __future__ import annotations

import re
import string
import pandas as pd
from tqdm import tqdm
from pyvi import ViTokenizer
from main.preprocessor.vocab.stopwords import STOP_WORDS
from main.preprocessor.vocab.title import TITLES
from main.preprocessor.vocab.vocab import ABBRE, SPECIAL_TERMS, ROMAN_NUM,LEGAL_WORDs,PROVINCES,CURRENCY
from main.preprocessor.utils.modify import (
    dupplicated_char_remover,
    preprocess_pyvi,
    postprocess_pyvi,
)
from main.preprocessor.utils.terms import terms_of_law
from main.preprocessor.utils.titles import titles_terms, ner_tokenize

__all__ = ["TextProcessor"]


class TextProcessor:
    def __init__(self):
        self.legal_term = LEGAL_WORDs
        self.stop_words = STOP_WORDS
        self.titles = TITLES
        self.special_terms = SPECIAL_TERMS

        # convenience lookup sets
        self._legal_tokens = set(self.legal_term.values())
        self._stopwords_tokens = {w for w in self.stop_words if w}
        self._titles_tokens = set(self.titles.values())
        self._special_tokens = set(self.special_terms)
    def _url_remover(self,paragraph):
        pattern = r"\([^)]*http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+[^)]*\)"

        def repl(match):
            content = match.group(0)
            cleaned = re.sub(r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+", "", content)
            return cleaned if cleaned.strip("() ") else ""
        
        return re.sub(pattern, repl, paragraph)

    def _punctuation_remover(self,paragraph):
        # Remove parentheses around numbers and certain patterns
        paragraph = re.sub(r"\((\d+)\)", r"\1", paragraph)
        paragraph = re.sub(r"\w+\)", " ", paragraph)

        words = paragraph.split()
        updated_words = []
        for item in words:
            if item.endswith((')', '.')) and item[0] in '123456789':
                continue
            if item in CURRENCY:
                updated_words.append(CURRENCY[item])
                continue
            for key, value in CURRENCY.items():
                if key in item:
                    item = item.replace(key, f" {value}")
                    break
            updated_words.append(item)
        paragraph = ' '.join(updated_words)

        for punc in string.punctuation:
            if punc == ":":
                paragraph = paragraph.replace(punc, ".")
            elif punc == '-':
                paragraph = paragraph.replace(punc, "")
            elif punc not in ["/", "."]:
                paragraph = paragraph.replace(punc, " ")
        return re.sub(r"\s+", " ", paragraph).strip()

    def _line_breaker_remover(self,paragraph):
        para = re.sub(r"\\n+", ". ", paragraph)
        para = re.sub(r"\n+", ". ", para)
        para = re.sub(r"\.\.\.", " ", para)
        para = re.sub(r'\.{1,}', '.', para)
        return para.replace("  ", " ")
    def _lowercase_standardizer(self,paragraph):
        return paragraph.lower()

    def _white_space_remover(self,paragraph):
        para = paragraph.replace("  ", " ")
        return re.sub(r"\s{2,}", " ", para).strip()

    def _legal_text_tokenizer(self, paragraph):
        for phrase, replacement in self.legal_term.items():
            paragraph = paragraph.replace(phrase, replacement)
        paragraph = terms_of_law(paragraph)
        paragraph = titles_terms(paragraph)
        paragraph = dupplicated_char_remover(paragraph)
        return paragraph

    def _text_tokenizer(self, paragraph):
        paragraph = ner_tokenize(paragraph)
        for phrase, replacement in self.special_terms.items():
            paragraph = paragraph.replace(phrase, replacement)
        paragraph = preprocess_pyvi(paragraph)
        paragraph = ViTokenizer.tokenize(paragraph)
        paragraph = postprocess_pyvi(paragraph)
        return paragraph

    def _stopword_remover(self, paragraph):
        return " ".join([w for w in paragraph.split() if w not in self.stop_words]).strip()

    def preprocess(
        self,
        docs,
        url_remover: bool = True,
        punctuation_remover: bool = True,
        line_breaker_remover: bool = True,
        lowercase_standardizer: bool = False,
        white_space_remover: bool = True,
        text_tokenizer: bool = True,
        law_text_recognizer: bool = True,
        stop_word_remover: bool = True):
        """Vectorised wrapper over *preprocess_text* (stage‑1)."""
        if isinstance(docs, pd.Series):
            tqdm.pandas(desc="Pre‑processing")
            return docs.progress_apply(
                lambda t: self.preprocess_text(
                    str(t),
                    url_remover,
                    punctuation_remover,
                    line_breaker_remover,
                    lowercase_standardizer,
                    white_space_remover,
                    text_tokenizer,
                    law_text_recognizer,
                    stop_word_remover,
                )
            )
        return self.preprocess_text(
            docs,
            url_remover,
            punctuation_remover,
            line_breaker_remover,
            lowercase_standardizer,
            white_space_remover,
            text_tokenizer,
            law_text_recognizer,
            stop_word_remover)

    def preprocess_text(self,
        paragraph: str,
        url_remover: bool = True,
        punctuation_remover: bool = True,
        line_breaker_remover: bool = True,
        lowercase_standardizer: bool = False,
        white_space_remover: bool = True,
        text_tokenizer: bool = True,
        law_text_recognizer: bool = True,
        stop_word_remover: bool = True):
        if url_remover:
            paragraph = self._url_remover(paragraph)
        if punctuation_remover:
            paragraph = self._punctuation_remover(paragraph)
        if line_breaker_remover:
            paragraph = self._line_breaker_remover(paragraph)
        if lowercase_standardizer:
            paragraph = self._lowercase_standardizer(paragraph)
        if white_space_remover:
            paragraph = self._white_space_remover(paragraph)
        if law_text_recognizer:
            paragraph = self._legal_text_tokenizer(paragraph)
        if text_tokenizer:
            paragraph = self._text_tokenizer(paragraph)
        if stop_word_remover:
            paragraph = self._stopword_remover(paragraph)
        return paragraph
    def _handle_n_items(self,text):
        numbers = map(str, range(1, 10))
        rewrite_text = []
        words = text.split("_")
        for word in words:
            if word.startswith('n'):
                if len(word) >= 2:
                    if word[1] == word[1].upper():
                        if any(num in word[1] for num in numbers):
                            rewrite_text.append(word[2:])
                        else:
                            rewrite_text.append(word[1:])
                    else:
                        rewrite_text.append(word)
                else:
                    rewrite_text.append(word)
            else:
                rewrite_text.append(word)
        return "_".join(rewrite_text)

    def _handle_rules(self,text: str):
        numbers = [str(i) for i in range(1, 1000)]
        rules = ['Khoản', 'Điều', 'Điểm', 'Chương', 'Cấp', 'Hạng', 'Mục']
        words = text.split("_")
        if len(words) == 4 and any(rule in text for rule in rules):
            return f"{words[0]}_{words[1]}", f"{words[-2]}_{words[-1]}"
        elif len(words) == 6 and any(rule in text for rule in rules):
            return (f"{words[0]}_{words[1]}", f"{words[2]}_{words[3]}", f"{words[-2]}_{words[-1]}")
        elif len(words) > 2 and words[-2] in rules and words[-1] in numbers:
            return (f"{words[-2]}_{words[-1]}", f"{words[:-2]}")
        else:
            return text

    def _handle_punc(self,text):
        text = text.strip(".")
        return text.replace(".", "")

    def _handle_rules_v2(self, text):
        for term in self._titles_tokens:
            if term in text:
                sub_text = text.split(term)[-1].strip("_")
                if (
                    sub_text in self._titles_tokens
                    or sub_text in self._special_tokens
                    or sub_text in self._legal_tokens
                ):
                    return term, sub_text
                return term
        return text

    def _handle_xa0_and_stopwords(self,text):
        new_text = text.replace('xa0', '').replace('xAA0', '')
        words = new_text.split("_")
        if words and words[-1].lower() in STOP_WORDS:
            new_text = "_".join(words[:-1])
        if len(words) > 2 and len(words) % 2 == 1 and 'trừ' in words[-1]:
            new_text = "_".join(words[:-1])
        return new_text
    def _handle_uppercase(self,text):
        if isinstance(text, (int, float)):
            return ""
        if text.isupper():
            words = text.lower().split("_")
            new_words = [
                w.lower() if any(c.isupper() for c in w[1:]) else w for w in words
            ]
            return "_".join(new_words)
        return text
    def _handle_abbre(self,text):
        return ABBRE.get(text, text)
    def _handle_numerical(self,text):
        try:
            return int(text)
        except ValueError:
            return text

    def _normalize_text(self, text):
        if any(p in text for p in PROVINCES) or \
           any(d in text for d in TITLES) or \
           any(r in text for r in ROMAN_NUM) or \
           any(k in text for k in ['Khoản', 'Điều', 'Điểm', 'Chương', 'Cấp', 'Hạng', 'Mục']):
            return text
        words = text.split('_')
        norm_words = [
            w.lower() if w not in PROVINCES and w not in TITLES and w not in ROMAN_NUM else w
            for w in words
        ]
        return '_'.join(norm_words)

    def post_preprocess_text(self, text):
        final_text= []
        text = self._handle_punc(text)
        text = self._handle_n_items(text)
        text = self._handle_xa0_and_stopwords(text)
        text = self._handle_numerical(text)
        text = self._handle_uppercase(text)
        text = self._handle_abbre(text)
        texts = self._handle_rules(text)
        if isinstance(texts, tuple):
            for segment in texts:
                seg = self._handle_rules_v2(segment)
                seg = self._normalize_text(seg)
                final_text.append(seg)
        else:
            texts = self._normalize_text(texts)
            final_text.append(texts)
        return final_text if len(final_text) > 1 else final_text[0]

    def post_preprocess(self, docs):
        if isinstance(docs, str):
            return self.post_preprocess_text(docs)
        if isinstance(docs, list):
            return [self.post_preprocess_text(doc) for doc in tqdm(docs, desc="Post‑processing")]
        tqdm.pandas(desc="Post‑processing")
        return docs.progress_apply(self.post_preprocess_text)
