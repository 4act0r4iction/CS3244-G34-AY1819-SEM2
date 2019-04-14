import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import nltk
from nltk.tokenize.toktok import ToktokTokenizer
import spacy
#>>>python3 -m spacy download en
spacy_nlp = spacy.load('en_core_web_sm')

#sklearn imports
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


from nltk.tokenize.toktok import ToktokTokenizer
stopword_list = spacy.lang.en.stop_words.STOP_WORDS
tokenizer = ToktokTokenizer()

import unicodedata
def remove_accented_chars(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text


from lib.contractions import CONTRACTION_MAP


def expand_contractions(text, contraction_mapping=CONTRACTION_MAP):
    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())),
                                      flags=re.IGNORECASE | re.DOTALL)

    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match) \
            if contraction_mapping.get(match) \
            else contraction_mapping.get(match.lower())
        expanded_contraction = first_char + expanded_contraction[1:]
        return expanded_contraction

    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text

import re

def remove_special_characters(text, remove_digits=False):
    pattern = r'[^a-zA-z0-9\s]' if not remove_digits else r'[^a-zA-z\s]'
    text = re.sub(pattern, '', text)
    return text

def simple_stemmer(text):
    ps = nltk.porter.PorterStemmer()
    text = ' '.join([ps.stem(word) for word in text.split()])
    return text

def lemmatize_text(text):
    text = spacy_nlp(text)
    text = ' '.join([word.lemma_ if word.lemma_ != '-PRON-' else word.text for word in text])
    return text

def remove_stopwords(text, is_lower_case=False):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopword_list]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text


def preprocess_text(text, remove_accents=True, remove_special =True, stem=True, lemmatize = True, remove_stops = True):
    if not text:
        return text

    if remove_accents:
        text = remove_accented_chars(text)
    if remove_special:
        text = remove_special_characters(text)
    if stem:
        text = simple_stemmer(text)
    if lemmatize:
        text = lemmatize_text(text)
    if remove_stops:
        text = remove_stopwords(text)
    return text