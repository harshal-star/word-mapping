import streamlit as st
import pandas as pd
import torch
from transformers import BertModel, BertTokenizer
from scipy import spatial

# Initialize BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def get_bert_embedding(word):
    inputs = tokenizer(word, return_tensors="pt")
    outputs = model(**inputs)
    # Use the mean of the last layer embeddings as the word representation
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.detach().numpy()[0]

def process_word(word):
    word = str(word).lower()
    if ":" in word:
        word = word.split(":")[1]
    return word.replace('-', '').replace(':', '')

def map_to_selected_word(selected_word, available_words, topk=5):
    selected_word_embedding = get_bert_embedding(process_word(selected_word))
    similarities = {}

    for word in available_words:
        word_embedding = get_bert_embedding(process_word(word))
        similarity = 1 - spatial.distance.cosine(selected_word_embedding, word_embedding)
        similarities[word] = similarity

    sorted_words = sorted(similarities, key=similarities.get, reverse=True)
    return sorted_words[:topk]

def parse_user_input(input_string):
    return [process_word(word) for word in input_string.split(',')]


# Streamlit app interface
st.title('Word Mapping Tool')

# User input for available words
user_input = st.text_area("Enter the available words separated by commas", "main:age_group,main:availability_date,main:brand,main:category")
# selected_word = st.selectbox("Select the word to map", options=wordlist)

# User input for the selected word
selected_word = st.text_input("Enter the word to map", "")


if st.button('Map Words'):
    available_words = parse_user_input(user_input)
    processed_selected_word = process_word(selected_word)
    if available_words and processed_selected_word:
        with st.spinner('Processing...'):
            mapped_words = map_to_selected_word(processed_selected_word, available_words, topk=5)
            st.success("Processing complete!")
            st.write("Top 5 closest words to", processed_selected_word, "are:", mapped_words)
    else:
        st.error("Please enter available words and a word to map before pressing 'Map Words'.")