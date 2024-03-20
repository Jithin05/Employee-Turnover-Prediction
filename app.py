from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load Universal Sentence Encoder
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

# Function to compute similarity score
def compute_similarity(text1, text2):
    # Encode text paragraphs
    embeddings = embed([text1, text2])

    # Compute cosine similarity
    similarity_score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    
    return similarity_score

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/compute_similarity', methods=['POST'])
def similarity():
    text1 = request.form['text1']
    text2 = request.form['text2']
    similarity_score = compute_similarity(text1, text2)
    return jsonify({"similarity_score": similarity_score})

if __name__ == '__main__':
    app.run(debug=True)
