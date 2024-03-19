import tensorflow as tf
import tensorflow_hub as hub
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

# Load Universal Sentence Encoder
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

# Function to compute similarity score
def compute_similarity(text1, text2):
    # Encode text paragraphs
    embeddings = embed([text1, text2])

    # Compute cosine similarity
    similarity_score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    
    return similarity_score

# Streamlit app UI
st.title("Semantic Textual Similarity")
text1 = st.text_area("Enter Text 1")
text2 = st.text_area("Enter Text 2")
if st.button("Compute Similarity"):
    similarity_score = compute_similarity(text1, text2)
    st.write("Similarity Score:", similarity_score)
