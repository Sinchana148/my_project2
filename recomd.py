from flask import Flask, request, render_template
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

app = Flask(__name__)

# Sample dataset
data = {
    'Movie': ['The Matrix', 'Inception', 'Interstellar', 'The Dark Knight', 'Avatar'],
    'Genre': ['Action Sci-Fi', 'Action Sci-Fi', 'Adventure Sci-Fi', 'Action Crime', 'Adventure Fantasy']
}
df = pd.DataFrame(data)

# Step 1: Create a Count Vectorizer
count_vectorizer = CountVectorizer()
genre_matrix = count_vectorizer.fit_transform(df['Genre'])

# Step 2: Calculate Cosine Similarity
cosine_sim = cosine_similarity(genre_matrix, genre_matrix)

# Recommendation Function
def recommend(movie_title):
    try:
        # Get the index of the given movie
        movie_idx = df[df['Movie'] == movie_title].index[0]

        # Get similarity scores for all movies
        similarity_scores = list(enumerate(cosine_sim[movie_idx]))

        # Sort the movies by similarity scores
        sorted_movies = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

        # Get top 3 recommendations
        recommendations = [df['Movie'][i[0]] for i in sorted_movies if i[0] != movie_idx][:3]
        return recommendations
    except IndexError:
        return ["Movie not found. Please check the name and try again."]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def get_recommendation():
    movie_title = request.form['movie']
    recommendations = recommend(movie_title)
    return render_template('index.html', recommendations=recommendations, movie_title=movie_title)

if __name__ == '_main_':
    app.run(debug=True)
