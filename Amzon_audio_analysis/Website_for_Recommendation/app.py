from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load data
df = pd.read_csv(r"C:\Users\ASUS\Amazon_Analysis_Project\cleaned_amazon_audio.csv")

# Create TF-IDF model
tfidf = TfidfVectorizer()
text_matrix = tfidf.fit_transform(df['product_title'])

def recommend_from_text(query, top_n=5, threshold=0.2):
    query_vec = tfidf.transform([query])
    sim_scores = cosine_similarity(query_vec, text_matrix).flatten()
    top_indices = sim_scores.argsort()[::-1]
    filtered = [i for i in top_indices if sim_scores[i] > threshold]

    if len(filtered) == 0:
        return []

    return df.iloc[filtered[:top_n]].to_dict(orient='records')

@app.route("/", methods=["GET", "POST"])
def home():
    results = []
    query = ""   # ✅ ADD THIS

    if request.method == "POST":
        query = request.form["query"]   # ✅ STORE QUERY
        results = recommend_from_text(query)

    return render_template("index.html", results=results, query=query)  # ✅ PASS QUERY

if __name__ == "__main__":
    app.run(debug=True)