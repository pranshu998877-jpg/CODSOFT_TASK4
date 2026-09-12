import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Create a mock movie dataset
movies_data = {
    'movie_id': [1, 2, 3, 4, 5, 6, 7],
    'title': ['The Dark Knight', 'Inception', 'Toy Story', 'Finding Nemo', 'Interstellar', 'Shrek', 'The Avengers'],
    'genres': ['Action Crime Drama', 'Action Sci-Fi Thriller', 'Animation Children Comedy', 'Animation Children Adventure', 'Sci-Fi Adventure Drama', 'Animation Children Comedy Fantasy', 'Action Sci-Fi Fantasy']
}

df = pd.DataFrame(movies_data)

def get_recommendations(movie_title, num_recommendations=2):
    # Ensure the movie exists in our mock data
    if movie_title not in df['title'].values:
        return f"Movie '{movie_title}' not found in the database."

    # 2. Compute the TF-IDF Vectorizer matrix for genres
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['genres'])

    # 3. Compute Cosine Similarity between all movies
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # 4. Fetch the index of the requested movie
    movie_idx = df[df['title'] == movie_title].index[0]

    # 5. Get pairwise similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[movie_idx]))

    # 6. Sort the movies based on similarity scores (highest first)
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # 7. Get the scores of the most similar items (excluding itself)
    sim_scores = sim_scores[1:num_recommendations + 1]

    # 8. Extract movie indices and return names
    recommended_indices = [i[0] for i in sim_scores]
    return df['title'].iloc[recommended_indices].tolist()

if __name__ == "__main__":
    print("--- Content-Based Movie Recommendation System ---")
    print("Available Movies:", df['title'].tolist())
    
    target_movie = "Inception"
    print(f"\nRecommendations if you liked '{target_movie}':")
    recommendations = get_recommendations(target_movie, num_recommendations=2)
    
    for idx, movie in enumerate(recommendations, 1):
        print(f"{idx}. {movie}")