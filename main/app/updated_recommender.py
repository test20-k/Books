
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
from fuzzywuzzy import fuzz
from tqdm import tqdm  # For progress bar during scoring
from sklearn.preprocessing import normalize # for better collab score
class OptimizedHybridRecommender:
    def __init__(self, cache_dir="D:\\Maybe\\model_cache"):
        self.cache_dir = Path(cache_dir)
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Initialize all required attributes
        self.books_data = None
        self.ratings_data = None
        self.tfidf = None
        self.svd = None
        self.reduced_matrix = None
        self.book_index = None  # Initialize here
        self.rating_pivot = None
        self.knn = None
        self.titles_list = []
        self.users_list = []

    def initialize(self, books_path, ratings_path, force_rebuild=False):
        """One-time initialization with data loading"""
        self.books_data, self.ratings_data = self.load_data(books_path, ratings_path)
        self.books_data = self.preprocess_text(self.books_data)
        self.train(books_path, ratings_path, force_rebuild=force_rebuild)

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

        # Normalize the rating pivot for improved cosine similarity
        rating_pivot_normalized = normalize(rating_pivot, axis=0)

        # Use approximate nearest neighbors
        knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20)
        knn.fit(rating_pivot_normalized) # Fit on the normalized matrix

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
        sample_indices = np.random.choice(reduced_matrix.shape[0], size=sample_size, replace=False)
        sample_matrix = reduced_matrix[sample_indices]

        # Compute cosine similarities
        similarities = cosine_similarity(target_vector, sample_matrix)[0]

        # Return top similar items
        top_indices = np.argsort(-similarities)[:n_neighbors]
        return [(sample_indices[i], similarities[i]) for i in top_indices if similarities[i] > 0.1]

    # def hybrid_recommend(self, user_id, top_n=5, alpha=0.5, candidate_size=1000):
    #     """Memory-efficient hybrid recommendation with index-safe scoring."""
    #     # Check if user exists
    #     user_exists = user_id in self.ratings_data['User_id'].values

    #     if not user_exists:
    #         if 'ratingsCount' in self.books_data.columns:
    #             return self.books_data.nlargest(top_n, 'ratingsCount')['Title'].tolist()
    #         return self.books_data.head(top_n)['Title'].tolist()

    #     user_rated = set(self.ratings_data[self.ratings_data['User_id'] == user_id]['Title'])
    #     candidate_books = self.books_data[~self.books_data['Title'].isin(user_rated)]['Title']

    #     if len(candidate_books) > candidate_size:
    #         candidate_books = np.random.choice(candidate_books, size=candidate_size, replace=False)

    #     user_high_rated = self.ratings_data[
    #         (self.ratings_data['User_id'] == user_id) &
    #         (self.ratings_data['review/score'] >= 4.0)
    #     ]['Title'].unique()

    #     scores = []
    #     for title in candidate_books:
    #         if title not in self.titles_list or title not in self.book_index:
    #             continue  # Skip books not in CF or content model

    #         # Collaborative filtering score
    #         try:
    #             cs = self.collab_score(user_id, title)
    #         except Exception:
    #             cs = 3.0

    #         # Content similarity score
    #         cb = 0
    #         if len(user_high_rated) > 0:
    #             similarities = []
    #             for hr_title in user_high_rated:
    #                 if hr_title in self.book_index and title in self.book_index:
    #                     try:
    #                         sim = cosine_similarity(
    #                             self.reduced_matrix[self.book_index[title]].reshape(1, -1),
    #                             self.reduced_matrix[self.book_index[hr_title]].reshape(1, -1)
    #                         )[0][0]
    #                         similarities.append(sim)
    #                     except Exception:
    #                         continue
    #             if similarities:
    #                 cb = np.mean(similarities)

    #         final_score = alpha * cs + (1 - alpha) * cb
    #         scores.a

    def get_weighted_content_similarity(self, user_id, target_title, high_rated_titles, n_neighbors=50):
        """
        Compute content similarity weighted by user's high ratings.
        """
        if target_title not in self.book_index:
            return 0

        target_idx = self.book_index[target_title]
        target_vector = self.reduced_matrix[target_idx].reshape(1, -1)

        sample_size = min(5000, self.reduced_matrix.shape[0])
        sample_indices = np.random.choice(self.reduced_matrix.shape[0], size=sample_size, replace=False)
        sample_matrix = self.reduced_matrix[sample_indices]

        similarities = cosine_similarity(target_vector, sample_matrix)[0]

        # Weight similarities by user's ratings for these books
        weights = np.zeros_like(similarities)
        user_ratings = self.ratings_data[self.ratings_data['User_id'] == user_id]
        for hr_title in high_rated_titles:
            if hr_title in self.book_index:
                hr_idx = self.book_index[hr_title]
                if hr_idx in sample_indices:
                    idx_in_sample = np.where(sample_indices == hr_idx)[0][0]
                    rating = user_ratings.loc[user_ratings['Title'] == hr_title, 'review/score'].values
                    if rating.size > 0:
                        weights[idx_in_sample] = rating[0]
        if weights.sum() > 0:
            weights /= weights.sum()  # normalize weights
        else:
            weights = np.ones_like(weights) / len(weights)

        # Compute weighted similarity
        weighted_similarity = np.dot(similarities, weights)

        # Get top N similar books
        top_idx = np.argsort(-similarities)[:n_neighbors]
        return [(self.books_data.iloc[sample_indices[i]]['Title'], similarities[i], weights[i]) for i in top_idx if similarities[i] > 0.1]

    def recommend(self, user_id, top_n=5, alpha=0.7):
        """
        Generate hybrid recommendations combining content and collaborative filtering.
        """
        user_rated_titles = set(self.ratings_data[self.ratings_data['User_id'] == user_id]['Title'])
        candidates = self.books_data[~self.books_data['Title'].isin(user_rated_titles)]['Title']

        # Find user's high-rated books
        high_rated_titles = self.ratings_data[
            (self.ratings_data['User_id'] == user_id) & (self.ratings_data['review/score'] >= 4)
        ]['Title'].values

        scores = []

        for title in candidates:
            # Content similarity weighted by user ratings
            content_score = get_weighted_content_similarity(self, user_id, title, high_rated_titles, n_neighbors=50)
            # Normalize content score from [-1, 1] to [0, 1]
            norm_content_score = (content_score + 1) / 2 if isinstance(content_score, float) else 0.5

            # Collaborative filtering score
            collab_score = None
            if user_id in self.ratings_data['User_id'].values and title in self.titles_list:
                collab_score = self.collab_score(user_id, title)
            else:
                collab_score = 3  # fallback

            # Normalize collab score from [1, 5] to [0, 1]
            norm_collab_score = (collab_score - 1) / 4

            # Combine with weight alpha
            final_score = alpha * norm_collab_score + (1 - alpha) * norm_content_score
            scores.append((title, final_score))

        # Return top N
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        return [t for t, s in scores[:top_n]]