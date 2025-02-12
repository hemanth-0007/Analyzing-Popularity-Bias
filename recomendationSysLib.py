import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from surprise import Dataset, Reader, SVD, KNNBasic
    from surprise.model_selection import cross_validate
    from collections import defaultdict
except ModuleNotFoundError:
    print("Error: The 'surprise' package is not installed. Please install it using 'pip install scikit-surprise'.")
    raise

# Load MovieLens 100K dataset
file_path = "https://files.grouplens.org/datasets/movielens/ml-100k/u.data"
df = pd.read_csv(file_path, sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])

# Drop timestamp column
df.drop(columns=['timestamp'], inplace=True)

# Exploratory Data Analysis
print("Dataset info:")
print(df.info())
print("\nSummary Statistics:")
print(df.describe())

# Distribution of Ratings
plt.figure(figsize=(8, 5))
sns.countplot(x=df['rating'], palette='viridis')
plt.title("Distribution of Ratings")
plt.xlabel("Rating")
plt.ylabel("Count")
plt.show()

# Load data into Surprise format
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[['user_id', 'item_id', 'rating']], reader)

# Collaborative Filtering Algorithms
algorithms = {
    'SVD': SVD(),
    'UserKNN': KNNBasic(sim_options={'user_based': True}),
    'ItemKNN': KNNBasic(sim_options={'user_based': False})
}

results = {}
for name, algo in algorithms.items():
    cv_results = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=False)
    results[name] = {
        'RMSE': np.mean(cv_results['test_rmse']),
        'MAE': np.mean(cv_results['test_mae'])
    }

# Convert results to DataFrame
results_df = pd.DataFrame(results).T
print("\nPerformance Metrics:")
print(results_df)

# Popularity Bias Analysis
popularity = df.groupby('item_id')['rating'].count().sort_values(ascending=False)
popular_items = set(popularity.head(100).index)

# Function to calculate how often popular items are recommended
def hit_rate(predictions, top_n=10):
    user_recs = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        user_recs[uid].append((iid, est))
    
    for uid in user_recs:
        user_recs[uid].sort(key=lambda x: x[1], reverse=True)
        user_recs[uid] = [iid for iid, _ in user_recs[uid][:top_n]]
    
    hits = sum(1 for recs in user_recs.values() if any(iid in popular_items for iid in recs))
    return hits / len(user_recs)

algo = SVD()
trainset = data.build_full_trainset()
algo.fit(trainset)
testset = trainset.build_testset()
predictions = algo.test(testset)

pop_bias = hit_rate(predictions)
print(f"\nPopularity Bias (Fraction of Top 100 Items Recommended): {pop_bias:.2f}")

# Visualization of Popularity Bias
labels = ["Popular Items Recommended", "Others"]
values = [pop_bias, 1 - pop_bias]
plt.figure(figsize=(6, 6))
plt.pie(values, labels=labels, autopct='%1.1f%%', colors=['blue', 'gray'])
plt.title("Popularity Bias in Recommendations")
plt.show()
