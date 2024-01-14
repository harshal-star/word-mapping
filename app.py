import streamlit as st
import fasttext
import fasttext.util
from scipy import spatial
import time
import pandas as pd

# Streamlit page configuration
st.set_page_config(layout="wide")

# Define stop characters
stop_characters = ['-', ':']

# feed_csv = "feed_dataset.csv"
# df = pd.read_csv(feed_csv, on_bad_lines='skip')
# wordlist = df['mapping_field'].tolist()
# wordlist = [str(word) for word in wordlist]

# Assuming the FastText model is loaded as shown in the previous examples
# @st.cache_data()
def load_fasttext_model():
    word_embedding_path = "C:/consult/Feedoptimize/Feedoptimize/compute_nearest/word_vectors/cc.en.300.bin"  # Update this path
    ft = fasttext.load_model(word_embedding_path)
    return ft

def remove_stop_characters(word):
    for c in stop_characters:
        word = word.replace(c,'')
    return word

# Function to remove stop characters and any prefixes ending with a colon
def process_word(word):
    word = str(word)
    word = word.lower()
    if(":" in word):
        word = word.split(":")[1]
    word = remove_stop_characters(word)
    return word

# Function to get the FastText embedding of a word
def get_fasttext_embedding(word):
    word_items = word.split('_')
    wordvec = sum(ft.get_word_vector(witem) for witem in word_items) / len(word_items)
    return wordvec

# Function to map a selected word to the available words
def map_to_selected_word(selected_word, available_words, topk = 5):
    processed_selected_word = process_word(selected_word)
    selected_word_vec = get_fasttext_embedding(processed_selected_word)
    # similarity_scores = {}
    cs_dict = {}

    for word in available_words:
        processed_word = process_word(word)
        word_vec = get_fasttext_embedding(processed_word)
        similarity = 1 - spatial.distance.cosine(selected_word_vec, word_vec)
        cs_dict[word] = similarity
    cs_dict = {k: v for k, v in sorted(cs_dict.items(), key=lambda item: item[1])}
    top_word_list = list(cs_dict.keys())[-topk:]
    top_word_list.reverse()
    # ret_list = [word + ", score: " + str(cs_dict[word]) for word in top_word_list]
    ret_list = top_word_list
    return ret_list
        # similarity_scores[processed_word] = similarity

    # Sorting the available words based on similarity scores
    sorted_words = sorted(similarity_scores, key=similarity_scores.get, reverse=True)
    return sorted_words

# Function to parse user input and return a list of available words without any prefixes
def parse_user_input(input_string):
    words_with_prefixes = input_string.split(',')
    available_words = [process_word(word) for word in words_with_prefixes]
    return available_words

# Streamlit app interface
st.title('Word Mapping Tool')

# User input for available words
user_input = st.text_area("Enter the available words separated by commas", "main:age_group,main:availability_date,main:brand,main:category")
# selected_word = st.selectbox("Select the word to map", options=wordlist)

# User input for the selected word
selected_word = st.text_input("Enter the word to map", "")

# Function to preprocess the selected_word input
def preprocess_selected_word(word):
    word = word.lower()  # Convert to lowercase
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
