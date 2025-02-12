import pandas as pd
import numpy as np
from surprise import Dataset, Reader, SVD, KNNBasic, NormalPredictor
from surprise.model_selection import cross_validate
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from surprise.model_selection import train_test_split

# Load the data
ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])
movies = pd.read_csv('ml-100k/u.item', sep='|', encoding='latin-1', 
                     names=['movie_id', 'title', 'release_date', 'video_release_date',
                           'IMDb_URL', 'unknown', 'Action', 'Adventure', 'Animation',
                           'Children', 'Comedy', 'Crime', 'Documentary', 'Drama',
                           'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery',
                           'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'])

# Basic data exploration
def explore_data(ratings_df, movies_df):
    print("Dataset Overview:")
    print(f"Number of users: {ratings_df['user_id'].nunique()}")
    print(f"Number of movies: {ratings_df['item_id'].nunique()}")
    print(f"Number of ratings: {len(ratings_df)}")
    print(f"\nRating distribution:")
    print(ratings_df['rating'].value_counts().sort_index())
    
    # Calculate rating density
    density = len(ratings_df) / (ratings_df['user_id'].nunique() * ratings_df['item_id'].nunique())
    print(f"\nRating density: {density:.4%}")
    
    return density

# Analyze popularity bias
def analyze_popularity_bias(ratings_df):
    item_popularity = ratings_df['item_id'].value_counts()
    
    # Calculate Gini coefficient
    pop_sorted = np.sort(item_popularity.values)
    n = len(pop_sorted)
    index = np.arange(1, n + 1)
    gini = ((2 * index - n - 1) * pop_sorted).sum() / (n * pop_sorted.sum())
    
    return item_popularity, gini

# Evaluate recommendations with popularity bias metrics
def evaluate_popularity_bias(predictions, item_popularity):
    # Calculate average popularity of recommended items
    item_recs = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        item_recs[uid].append((iid, est))
    
    avg_pop = []
    for uid, recs in item_recs.items():
        # Sort by estimated rating and get top 10
        top_recs = sorted(recs, key=lambda x: x[1], reverse=True)[:10]
        # Get popularity of recommended items
        rec_pop = [item_popularity.get(iid, 0) for iid, _ in top_recs]
        avg_pop.append(np.mean(rec_pop))
    
    return np.mean(avg_pop)

# Main analysis
# Data exploration
density = explore_data(ratings, movies)

# Create Surprise reader and data
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings[['user_id', 'item_id', 'rating']], reader)

# Split data
trainset, testset = train_test_split(data, test_size=0.25, random_state=42)

# Analyze popularity bias
item_popularity, gini_coefficient = analyze_popularity_bias(ratings)

# Initialize algorithms
algorithms = {
    'SVD': SVD(random_state=42),
    'KNN': KNNBasic(sim_options={'user_based': True}),
    'Random': NormalPredictor(random_state=42)
}

# Results storage
results = []

# Evaluate each algorithm
for name, algo in algorithms.items():
    # Train and predict
    algo.fit(trainset)
    predictions = algo.test(testset)
    
    # Calculate metrics
    avg_popularity = evaluate_popularity_bias(predictions, item_popularity)
    cv_results = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=False)
    
    results.append({
        'Algorithm': name,
        'RMSE': cv_results['test_rmse'].mean(),
        'MAE': cv_results['test_mae'].mean(),
        'Avg_Rec_Popularity': avg_popularity
    })

# Create results table
results_df = pd.DataFrame(results)

# Plotting results
plt.figure(figsize=(12, 6))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Plot RMSE and MAE
results_df.plot(x='Algorithm', y=['RMSE', 'MAE'], kind='bar', ax=ax1)
ax1.set_title('RMSE and MAE by Algorithm')
ax1.set_ylabel('Error')
ax1.legend(loc='upper right')

# Plot Average Recommendation Popularity
results_df.plot(x='Algorithm', y='Avg_Rec_Popularity', kind='bar', ax=ax2)
ax2.set_title('Average Recommendation Popularity by Algorithm')
ax2.set_ylabel('Average Popularity')

plt.tight_layout()
plt.show()

# Print detailed results
print("\nDetailed Results:")
print(results_df.round(4).to_string(index=False))
print(f"\nGini Coefficient (Popularity Concentration): {gini_coefficient:.4f}")