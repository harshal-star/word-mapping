import streamlit as st
import fasttext
import boto3
from io import BytesIO
from scipy import spatial
import pandas as pd
import os
import tempfile
import smart_open
from pathlib import Path
import time
import compress_fasttext
import os
from dotenv import load_dotenv
import numpy as np

load_dotenv()

# Set these in your environment or in your code before calling smart_open
os.environ['AWS_ACCESS_KEY_ID'] = st.secrets["AWS_ACCESS_KEY_ID"]
os.environ['AWS_SECRET_ACCESS_KEY'] = st.secrets["AWS_SECRET_ACCESS_KEY"]

# Streamlit page configuration
st.set_page_config(layout="wide")

# Define stop characters
stop_characters = ['-', ':']


def load_fasttext_model():
    file = smart_open.smart_open(f's3://wordmappingbucket/compressed.bin')
    listed = b''.join([i for i in file])
    with tempfile.TemporaryDirectory() as tdir:
        tfile = Path(tdir).joinpath('tempfile.bin')
        tfile.write_bytes(listed)
        ft = compress_fasttext.models.CompressedFastTextKeyedVectors.load(str(tfile))
    return ft


def remove_stop_characters(word):
    for c in stop_characters:
        word = word.replace(c,'')
    return word

def process_word(word):
    word = str(word)
    word = word.lower()
    if(":" in word):
        word = word.split(":")[1]
    word = remove_stop_characters(word)
    return word

# Function to get the FastText embedding of a word
def get_fasttext_embedding(ft, word):
    word_items = word.split('_')
    wordvec = sum(ft.word_vec(witem) for witem in word_items) / len(word_items)
    return wordvec

def map_to_selected_word(selected_word, available_words, topk=5):
    processed_selected_word = process_word(selected_word)
    selected_word_vec = get_fasttext_embedding(ft, processed_selected_word)

    # Check if selected_word_vec is a zero vector
    if np.linalg.norm(selected_word_vec) == 0:
        return ["Selected word vector is zero. Cannot compute similarity."]

    cs_dict = {}

    for word in available_words:
        processed_word = process_word(word)
        word_vec = get_fasttext_embedding(ft, processed_word)

        # Check if word_vec is a zero vector
        if np.linalg.norm(word_vec) == 0:
            continue

        similarity = 1 - spatial.distance.cosine(selected_word_vec, word_vec)

        # Skip invalid similarity scores (like NaN)
        if np.isnan(similarity):
            continue

        cs_dict[word] = similarity

    # Sort the dictionary by similarity score in descending order
    cs_dict = {k: v for k, v in sorted(cs_dict.items(), key=lambda item: item[1], reverse=True)}

    # Get top k words
    top_word_list = list(cs_dict.keys())[:topk]

    return top_word_list

def parse_user_input(input_string):
    words_with_prefixes = input_string.split(',')
    available_words = [process_word(word) for word in words_with_prefixes]
    return available_words

st.title('Word Mapping Tool')

user_input = st.text_area("Enter the available words separated by commas", "main:age_group,main:availability_date,main:brand,main:category")
selected_word = st.text_input("Enter the word to map", "")

def preprocess_selected_word(word):
    word = word.lower()
    words = word.split()
    return "_".join(words) if len(words) > 1 else word


# Button to perform the mapping
if st.button('Map Words'):
    ft = load_fasttext_model()
    # Parsing user input for available words
    available_words = parse_user_input(user_input) if user_input else []
    # Preprocessing the user input for selected_word
    processed_selected_word = preprocess_selected_word(selected_word)
    # Ensuring there is an available word and a user-entered word before proceeding
    if available_words and processed_selected_word:
        with st.spinner('Processing...'):
            # Perform the mapping
            mapped_words = map_to_selected_word(processed_selected_word, available_words, topk=1)
            st.success("Processing complete!")
            st.write("The closest word to", processed_selected_word, "is:", mapped_words[0])
    else:
        st.error("Please enter available words and a word to map before pressing 'Map Words'.")
