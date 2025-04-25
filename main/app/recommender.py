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
        sample_indices = np.random.choice(reduced_matrix.shape[0], size=sample_size, replace=False)
        sample_matrix = reduced_matrix[sample_indices]

        # Compute cosine similarities
        similarities = cosine_similarity(target_vector, sample_matrix)[0]

        # Return top similar items
        top_indices = np.argsort(-similarities)[:n_neighbors]
        return [(sample_indices[i], similarities[i]) for i in top_indices if similarities[i] > 0.1]

    def hybrid_recommend(self, user_id, top_n=5, alpha=0.5, candidate_size=1000):
        """Memory-efficient hybrid recommendation with index-safe scoring."""
        # Check if user exists
        user_exists = user_id in self.ratings_data['User_id'].values

        if not user_exists:
            if 'ratingsCount' in self.books_data.columns:
                return self.books_data.nlargest(top_n, 'ratingsCount')['Title'].tolist()
            return self.books_data.head(top_n)['Title'].tolist()

        user_rated = set(self.ratings_data[self.ratings_data['User_id'] == user_id]['Title'])
        candidate_books = self.books_data[~self.books_data['Title'].isin(user_rated)]['Title']

        if len(candidate_books) > candidate_size:
            candidate_books = np.random.choice(candidate_books, size=candidate_size, replace=False)

        user_high_rated = self.ratings_data[
            (self.ratings_data['User_id'] == user_id) &
            (self.ratings_data['review/score'] >= 4.0)
        ]['Title'].unique()

        scores = []
        for title in candidate_books:
            if title not in self.titles_list or title not in self.book_index:
                continue  # Skip books not in CF or content model

            # Collaborative filtering score
            try:
                cs = self.collab_score(user_id, title)
            except Exception:
                cs = 3.0

            # Content similarity score
            cb = 0
            if len(user_high_rated) > 0:
                similarities = []
                for hr_title in user_high_rated:
                    if hr_title in self.book_index and title in self.book_index:
                        try:
                            sim = cosine_similarity(
                                self.reduced_matrix[self.book_index[title]].reshape(1, -1),
                                self.reduced_matrix[self.book_index[hr_title]].reshape(1, -1)
                            )[0][0]
                            similarities.append(sim)
                        except Exception:
                            continue
                if similarities:
                    cb = np.mean(similarities)

            final_score = alpha * cs + (1 - alpha) * cb
            scores.append((title, final_score))

        return [title for title, score in sorted(scores, key=lambda x: -x[1])[:top_n]]



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

        # Ensure indices are within valid range
        if title_idx < 0 or title_idx >= self.rating_pivot.shape[0]:
            print(f"Invalid title index: {title_idx}")
            return 3.0
        if user_idx < 0 or user_idx >= self.rating_pivot.shape[1]:
            print(f"Invalid user index: {user_idx}")
            return 3.0

        distances, indices = self.knn.kneighbors(
            self.rating_pivot[title_idx],
            n_neighbors=20
        )

        user_ratings = self.rating_pivot[:, user_idx].toarray().flatten()

        mask = (distances[0] < 0.999) & (user_ratings[indices[0]] > 0)
        valid_indices = indices[0][mask]
        valid_distances = distances[0][mask]

        if len(valid_indices) == 0:
            return user_ratings.mean() if user_ratings.sum() > 0 else 3.0

        neighbor_ratings = user_ratings[valid_indices]
        neighbor_similarities = 1 - valid_distances

        if neighbor_similarities.sum() == 0:
            return neighbor_ratings.mean()

        weighted_sum = np.dot(neighbor_ratings, neighbor_similarities)
        normalized_score = weighted_sum / neighbor_similarities.sum()

        return normalized_score

    def load_models(self):
        """Load pre-trained content and collaborative models from cache"""
        print("Loading pre-trained models...")

        # Load content-based models
        with open(self.cache_dir / "tfidf.pkl", 'rb') as f:
            self.tfidf = pickle.load(f)

        with open(self.cache_dir / "svd.pkl", 'rb') as f:
            self.svd = pickle.load(f)

        self.reduced_matrix = load_npz(self.cache_dir / "reduced_matrix.npz")

        with open(self.cache_dir / "book_index.pkl", 'rb') as f:
            self.book_index = pickle.load(f)  # <-- fixed: use normal pickle, not pandas

        # Load collaborative filtering models
        self.rating_pivot = load_npz(self.cache_dir / "rating_pivot.npz")

        self.knn = pd.read_pickle(self.cache_dir / "knn_model.pkl")

        with open(self.cache_dir / "cf_metadata.pkl", 'rb') as f:
            metadata = pickle.load(f)
            self.titles_list = metadata['titles_list']
            self.users_list = metadata['users_list']

        print("Models loaded successfully.")

    def get_combined_score(self, user_id, book_title, alpha=0.5):
        """
        Computes a combined score from content-based and collaborative filtering.
        If the user is new (not found in ratings), returns only content-based score.
        """
        # Check whether the book exists in the index
        if book_title not in self.book_index:
            print(f"Book '{book_title}' not found in content index.")
            return None

        # Compute content-based score (similarity with user's highly rated books)
        user_exists = user_id in self.ratings_data['User_id'].values
        content_score = None

        if user_exists:
            user_high_rated = self.ratings_data[
                (self.ratings_data['User_id'] == user_id) &
                (self.ratings_data['review/score'] >= 4.0)
            ]['Title'].unique()

            similarities = []
            for hr_title in user_high_rated:
                if hr_title in self.book_index:
                    sim = cosine_similarity(
                        self.reduced_matrix[self.book_index[book_title]].reshape(1, -1),
                        self.reduced_matrix[self.book_index[hr_title]].reshape(1, -1)
                    )[0][0]
                    similarities.append(sim)

            if similarities:
                content_score = np.mean(similarities)
            else:
                content_score = 0  # No high-rated books found, default similarity
        else:
            # New user: no ratings history, default to general content similarity
            content_score = 0.5  # neutral default score

        # Compute collaborative filtering score
        if user_exists and book_title in self.titles_list:
            collab_score = self.collab_score(user_id, book_title)
        else:
            collab_score = None  # No collaborative information available

        # Combine scores appropriately
        if collab_score is not None:
            final_score = alpha * collab_score + (1 - alpha) * content_score
        else:
            final_score = content_score  # fallback purely content-based for new users

        return {
            'book_title': book_title,
            'content_score': content_score,
            'collaborative_score': collab_score,
            'final_score': final_score
        }
    def recommend_from_combined_score(self, user_id, top_n=5, alpha=0.5, candidate_size=1000):
        """Generate recommendations by scoring all candidate books using get_combined_score."""
        # Check if user exists
        user_exists = user_id in self.ratings_data['User_id'].values

        # Get books the user has already rated
        if user_exists:
            rated_books = set(self.ratings_data[self.ratings_data['User_id'] == user_id]['Title'])
        else:
            rated_books = set()

        # Candidate books (not yet rated by the user)
        candidate_books = self.books_data[~self.books_data['Title'].isin(rated_books)]['Title'].unique()

        # If too many candidates, randomly sample
        if len(candidate_books) > candidate_size:
            candidate_books = np.random.choice(candidate_books, size=candidate_size, replace=False)

        scored_books = []

        for title in candidate_books:
            matched_title = find_title(title, self.book_index)  # fix casing / spacing
            if matched_title is None:
                continue

            score_info = self.get_combined_score(user_id, matched_title, alpha=alpha)
            if score_info is not None:
                scored_books.append((score_info['book_title'], score_info['final_score']))

        # Sort by final_score descending
        scored_books = sorted(scored_books, key=lambda x: -x[1])

        # Return top_n book titles
        return [title for title, score in scored_books[:top_n]]
    
    def hybrid_recommend_based_on_title(self, user_id, book_title, top_n=5, n_neighbors=50):
        """Recommend books with proper score extraction"""
        matched_title = find_title(book_title, self.book_index)
        if matched_title is None:
            print(f"Book '{book_title}' not found in index.")
            return []

        try:
            idx = self.book_index[matched_title]
            target_vector = self.reduced_matrix[idx].reshape(1, -1)
            similarities = cosine_similarity(target_vector, self.reduced_matrix)[0]
            
            similar_indices = similarities.argsort()[::-1][1:n_neighbors+1]
            candidate_books = [self.books_data.iloc[i]['Title'] for i in similar_indices]

            scored_candidates = []
            for candidate in candidate_books:
                score_info = self.get_combined_score(user_id, candidate)
                if score_info and 'final_score' in score_info:
                    scored_candidates.append(
                        (candidate, score_info['final_score'])
                    )

            # Sort by numerical score descending
            scored_candidates.sort(key=lambda x: x[1], reverse=True)
            
            return [book for book, score in scored_candidates[:top_n]]

        except Exception as e:
            print(f"Recommendation error: {str(e)}")
            return self.books_data.sample(n=top_n)['Title'].tolist()
   


recommender = OptimizedHybridRecommender()

# # Instead of .train() again
recommender.books_data, recommender.ratings_data = recommender.load_data('D:\\End_Semester\\recommender\\data\\cleaned_books.csv', 'D:\\End_Semester\\recommender\\data\\cleaned_ratings.csv')
recommender.books_data = recommender.preprocess_text(recommender.books_data)
recommender.load_models()

# # # Now you can call
# # recommendations = recommender.hybrid_recommend(user_id='12345', top_n=5)
# # print(recommendations)

# # Example book title and user ID
# # user_id = '12345'
# # book_title = 'Harry Potter and the Sorcerer\'s Stone'

# # # Get combined recommendation score
# # result = recommender.get_combined_score(user_id, "Harry Potter and The Sorcerer's Stone", alpha=0.6)

# # print(f"Results for user '{user_id}' and book '{book_title}':")
# # print(result)

def find_title(title, index):
    t = title.lower().strip()
    for k in index.keys():
        if k.lower().strip() == t:
            return k
    return None

# # book_title = 'Harry Potter and the Sorcerer\'s Stone'
# # matched = find_title(book_title, recommender.book_index)
# # print("Matched title:", matched)

result=recommender.hybrid_recommend_based_on_title(user_id="jugh", book_title="Harry Potter and The Sorcerer's Stone", top_n=5)
print("Matched title:", result)
result=recommender.improved_hybrid_recommend(user_id="jugh", top_n=5)
print("Matched title:", result)
result=recommender.hybrid_recommend_based_on(user_id="jugh",book_title="Harry Potter and The Sorcerer's Stone", top_n=5)
print("Matched title:", result)
result=recommender.hybrid_recommend(user_id="A002258237PFYJV336T05", top_n=5)
print("Matched title:", result)
# import os
# import shutil
# cache_dir = "D:\\End_Semester\\recommender\\data\\model_cache\\model_cache"
# shutil.rmtree(cache_dir)  # Deletes the entire cache directory
# os.makedirs(cache_dir, exist_ok=True)
# recommender = OptimizedHybridRecommender(cache_dir=cache_dir)

# # Fully retrain (force_rebuild=True ensures fresh training)
recommender.train(
    books_path='D:\\End_Semester\\recommender\\data\\cleaned_books.csv',
    ratings_path='D:\\End_Semester\\recommender\\data\\cleaned_ratings.csv',
    sample_frac=1.0,
    force_rebuild=True
)