import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix, save_npz, load_npz
import os
from pathlib import Path
import pickle
from sklearn.metrics.pairwise import cosine_similarity

class OptimizedHybridRecommender:
    def __init__(self, cache_dir="model_cache"):
        self.cache_dir = Path(cache_dir)
        os.makedirs(self.cache_dir, exist_ok=True)

    def load_data(self, books_path, ratings_path, sample_frac=1.0):
        """Load and optionally sample data"""
        print("Loading data...")
        books = pd.read_csv(books_path)
        ratings = pd.read_csv(ratings_path)

        if sample_frac < 1.0:
            ratings = ratings.sample(frac=sample_frac, random_state=42)
            books = books[books['Title'].isin(ratings['Title'])]

        # Basic cleaning
        books = books[books['Title'].notna()]
        ratings = ratings.dropna(subset=['review/score'])
        ratings['review/score'] = ratings['review/score'].astype(float)

        return books, ratings

    def preprocess_text(self, books_data):
        """Prepare text features"""
        print("Preprocessing text...")
        books_data['combined_text'] = (
            books_data['Title'].fillna('') + ' ' +
            books_data['description'].fillna('') + ' ' +
            books_data['authors'].fillna('') + ' ' +
            books_data['categories'].fillna('')
        )
        return books_data

    def build_content_model(self, books_data, n_components=300, force_rebuild=False):
        """Build content model with dimensionality reduction"""
        cache_files = {
            'tfidf': self.cache_dir / "tfidf.pkl",
            'svd': self.cache_dir / "svd.pkl",
            'reduced_matrix': self.cache_dir / "reduced_matrix.npz",
            'book_index': self.cache_dir / "book_index.pkl"
        }

        if not force_rebuild and all(f.exists() for f in cache_files.values()):
            print("Loading cached content models...")
            with open(cache_files['tfidf'], 'rb') as f:
                tfidf = pickle.load(f)
            with open(cache_files['svd'], 'rb') as f:
                svd = pickle.load(f)
            reduced_matrix = load_npz(cache_files['reduced_matrix'])
            book_index = pd.read_pickle(cache_files['book_index'])
            return tfidf, svd, reduced_matrix, book_index

        print("Building content models...")

        # TF-IDF with reduced features
        tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
        tfidf_matrix = tfidf.fit_transform(books_data['combined_text'])

        # Dimensionality reduction
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        reduced_matrix = svd.fit_transform(tfidf_matrix)

        # Convert to sparse and save
        reduced_matrix_sparse = csr_matrix(reduced_matrix)
        book_index = pd.Series(books_data.index, index=books_data['Title']).drop_duplicates()

        # Cache models
        with open(cache_files['tfidf'], 'wb') as f:
            pickle.dump(tfidf, f)
        with open(cache_files['svd'], 'wb') as f:
            pickle.dump(svd, f)
        save_npz(cache_files['reduced_matrix'], reduced_matrix_sparse)
        book_index.to_pickle(cache_files['book_index'])

        return tfidf, svd, reduced_matrix_sparse, book_index

    def build_collab_model(self, ratings_data, force_rebuild=False):
        """Build collaborative model with sparse matrices"""
        cache_files = {
            'pivot': self.cache_dir / "rating_pivot.npz",
            'knn': self.cache_dir / "knn_model.pkl",
            'metadata': self.cache_dir / "cf_metadata.pkl"
        }

        if not force_rebuild and all(f.exists() for f in cache_files.values()):
            print("Loading cached collaborative models...")
            rating_pivot = load_npz(cache_files['pivot'])
            knn = pd.read_pickle(cache_files['knn'])
            with open(cache_files['metadata'], 'rb') as f:
                metadata = pickle.load(f)
                titles_list = metadata['titles_list']
                users_list = metadata['users_list']
            return rating_pivot, knn, titles_list, users_list

        print("Building collaborative models...")

        # Create sparse user-item matrix
        unique_titles = ratings_data['Title'].unique()
        unique_users = ratings_data['User_id'].unique()

        title_to_idx = {title: idx for idx, title in enumerate(unique_titles)}
        user_to_idx = {user: idx for idx, user in enumerate(unique_users)}

        row_indices = ratings_data['Title'].map(title_to_idx)
        col_indices = ratings_data['User_id'].map(user_to_idx)

        rating_pivot = csr_matrix(
            (ratings_data['review/score'], (row_indices, col_indices)),
            shape=(len(unique_titles), len(unique_users))
        )

        # Use approximate nearest neighbors
        knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20)
        knn.fit(rating_pivot)

        # Save metadata
        metadata = {
            'titles_list': unique_titles.tolist(),
            'users_list': unique_users.tolist()
        }

        # Cache models
        save_npz(cache_files['pivot'], rating_pivot)
        pd.to_pickle(knn, cache_files['knn'])
        with open(cache_files['metadata'], 'wb') as f:
            pickle.dump(metadata, f)

        return rating_pivot, knn, unique_titles.tolist(), unique_users.tolist()

    def get_content_score(self, book_title, reduced_matrix, book_index, n_neighbors=50):
        """Compute content similarity on-demand without full matrix"""
        if book_title not in book_index:
            return None

        idx = book_index[book_title]
        target_vector = reduced_matrix[idx].reshape(1, -1)

        # Compute similarity to a sample of other books (or all if small enough)
        sample_size = min(5000, reduced_matrix.shape[0])  # Adjust based on memory
        all_book_indices = self.books_data.index.to_numpy() # Get the actual index values
        sampled_book_indices = np.random.choice(all_book_indices, size=sample_size, replace=False)
        sample_matrix_indices = [self.book_index[self.books_data.loc[book_idx, 'Title']] for book_idx in sampled_book_indices if self.books_data.loc[book_idx, 'Title'] in self.book_index]
        sample_matrix = reduced_matrix[sample_matrix_indices]

        # Compute cosine similarities
        similarities = cosine_similarity(target_vector, sample_matrix)[0]

        # Get the titles of the sampled books
        sampled_titles = self.books_data.loc[sampled_book_indices, 'Title'].tolist()
        title_similarity_pairs = list(zip(sampled_titles, similarities))

        # Sort by similarity and return top similar items
        sorted_pairs = sorted(title_similarity_pairs, key=lambda item: item[1], reverse=True)
        top_recommendations = [(title, score) for title, score in sorted_pairs if score > 0.1][:n_neighbors]

        return top_recommendations

    def hybrid_recommend(self, user_id=None, book_title=None, top_n=5, alpha=0.5, candidate_size=1000):
        """Hybrid recommendation based on user or book."""
        if book_title:
            if book_title not in self.book_index:
                print(f"Book '{book_title}' not found in the model.")
                return []
            similar_books_with_scores = self.get_content_score(
                book_title, self.reduced_matrix, self.book_index, n_neighbors=top_n + 1
            )
            return [title for title, score in similar_books_with_scores if title != book_title][:top_n]

        elif user_id:
            user_exists = user_id in self.ratings_data['User_id'].values

            if not user_exists:
                if 'ratingsCount' in self.books_data.columns:
                    return self.books_data.nlargest(top_n, 'ratingsCount')['Title'].tolist()
                return self.books_data.head(top_n)['Title'].tolist()

            user_rated = set(self.ratings_data[self.ratings_data['User_id'] == user_id]['Title'])
            candidate_books_cf = self.books_data[~self.books_data['Title'].isin(user_rated)]['Title']

            if len(candidate_books_cf) > candidate_size:
                candidate_books_cf = np.random.choice(candidate_books_cf, size=candidate_size, replace=False)

            user_high_rated = self.ratings_data[
                (self.ratings_data['User_id'] == user_id) &
                (self.ratings_data['review/score'] >= 4.0)
            ]['Title'].unique()

            hybrid_scores = {}

            # Collaborative Filtering Scores
            for title in candidate_books_cf:
                if title in self.titles_list:
                    try:
                        cf_score = self.collab_score(user_id, title)
                        hybrid_scores[title] = hybrid_scores.get(title, 0) + alpha * cf_score
                    except Exception:
                        pass

            # Content-Based Scores (based on user's liked books)
            for liked_book in user_high_rated:
                if liked_book in self.book_index:
                    similar_books = self.get_content_score(liked_book, self.reduced_matrix, self.book_index, n_neighbors=20)
                    if similar_books:
                        for book, similarity in similar_books:
                            if book not in user_rated:
                                hybrid_scores[book] = hybrid_scores.get(book, 0) + (1 - alpha) * similarity

            sorted_recommendations = sorted(hybrid_scores.items(), key=lambda item: item[1], reverse=True)[:top_n]
            return [title for title, score in sorted_recommendations]

        else:
            print("Please provide either a user_id or a book_title for recommendations.")
            return []

    def recommend_based_on_book(self, book_title, top_n=5):
        """Generate recommendations based on a given book title (content-based)."""
        if book_title not in self.book_index:
            print(f"Book '{book_title}' not found in the model.")
            return []

        similar_books_with_scores = self.get_content_score(
            book_title, self.reduced_matrix, self.book_index, n_neighbors=top_n + 1
        )

        if not similar_books_with_scores:
            print(f"No similar books found for '{book_title}'.")
            return []

        recommendations = [
            title for title, score in similar_books_with_scores if title != book_title
        ][:top_n]

        return recommendations

    def train(self, books_path, ratings_path, sample_frac=1.0, force_rebuild=False):
        """Train the complete system"""
        self.books_data, self.ratings_data = self.load_data(books_path, ratings_path, sample_frac)
        self.books_data = self.preprocess_text(self.books_data)

        # Build models
        self.tfidf, self.svd, self.reduced_matrix, self.book_index = \
            self.build_content_model(self.books_data, force_rebuild=force_rebuild)

        self.rating_pivot, self.knn, self.titles_list, self.users_list = \
            self.build_collab_model(self.ratings_data, force_rebuild=force_rebuild)

        print("Training completed successfully")

    def collab_score(self, user_id, title):
        """Compute collaborative filtering score"""
        try:
            user_idx = self.users_list.index(user_id)
            title_idx = self.titles_list.index(title)
        except ValueError:
            return 3.0  # Fallback rating if user or title is not found

        # Ensure indices are within valid range before accessing the matrix
        if title_idx < 0 or title_idx >= self.rating_pivot.shape[0]:
            print(f"Invalid title index: {title_idx}")
            return 3.0
        if user_idx < 0 or user_idx >= self.rating_pivot.shape[1]:
            print(f"Invalid user index: {user_idx}")
            return 3.0

        # Find nearest neighbors
        distances, indices = self.knn.kneighbors(
            self.rating_pivot[title_idx],
            n_neighbors=20
        )

        # Get user's ratings
        user_ratings = self.rating_pivot[:, user_idx].toarray().flatten()

        # Filter valid neighbors
        mask = (distances[0] < 0.999) & (user_ratings[indices[0]] > 0)
        valid_indices = indices[0][mask]
        valid_distances = distances[0][mask]

        if len(valid_indices) == 0:
            return user_ratings.mean() if user_ratings.sum() > 0 else 3.0

        # Weighted average
        similarities = 1 - valid_distances
        neighbor_ratings = user_ratings[valid_indices]
        return np.dot(neighbor_ratings, similarities) / similarities.sum()

if __name__ == "__main__":
    recommender = OptimizedHybridRecommender()
    cache_files_exist = [
        (recommender.cache_dir / "tfidf.pkl").exists(),
        (recommender.cache_dir / "svd.pkl").exists(),
        (recommender.cache_dir / "reduced_matrix.npz").exists(),
        (recommender.cache_dir / "book_index.pkl").exists(),
        (recommender.cache_dir / "rating_pivot.npz").exists(),
        (recommender.cache_dir / "knn_model.pkl").exists(),
        (recommender.cache_dir / "cf_metadata.pkl").exists()
    ]
    cache_exists = all(cache_files_exist)

    if cache_exists:
        print("Loading pre-trained models...")
        recommender.train(
            books_path= 'D:\\End_Semester\\recommender\\data\\cleaned_books.csv',
            ratings_path='D:\\End_Semester\\recommender\\data\\cleaned_ratings.csv',
            sample_frac=0.1,
            force_rebuild=False
        )
    else:
        print("No pre-trained models found. Starting training...")
        recommender.train(
            books_path= 'D:\\End_Semester\\recommender\\data\\cleaned_books.csv',
            ratings_path='D:\\End_Semester\\recommender\\data\\cleaned_ratings.csv',
            sample_frac=0.1,
            force_rebuild=True
        )

    user_id = "A30TK6U7DNS82R"
    user_recommendations = recommender.hybrid_recommend(user_id=user_id, top_n=5, alpha=0.6)
    print(f"Recommendations for user {user_id}: {user_recommendations}")

    book_title = "The Alchemist"  # Example book title
    book_based_recommendations = recommender.recommend_based_on_book(book_title, top_n=5)
    print(f"Recommendations based on '{book_title}': {book_based_recommendations}")

    book_based_recommendations_hybrid = recommender.hybrid_recommend(user_id=user_id,book_title=book_title, top_n=5)
    print(f"Hybrid recommendations based on '{book_title}': {book_based_recommendations_hybrid}")