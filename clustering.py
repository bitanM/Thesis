import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# CONFIGURATION
# ==========================================
CSV_FILE = r'C:\Users\HP-PC\Desktop\Social Media\IranIsrael\IranIsrael.csv'      # Your file name
COLUMN_NAME = 'action_tweet_text'         # The column header containing the tweets
MIN_K = 3                    # Minimum clusters to test
MAX_K = 10                   # Maximum clusters to test
# ==========================================

# 1. Load Data
print("Loading data...")
try:
    df = pd.read_csv(CSV_FILE)
except FileNotFoundError:
    print(f"Error: File {CSV_FILE} not found.")
    exit()

if COLUMN_NAME not in df.columns:
    print(f"Error: Column '{COLUMN_NAME}' not found.")
    exit()

# 2. Preprocessing
print("Preprocessing tweets...")
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

def clean_tweet(text):
    if not isinstance(text, str): return []
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    # Filter words > 2 chars and not in stopwords
    tokens = [word for word in text.split() if word not in stop_words and len(word) > 2]
    return tokens

tokens_list = df[COLUMN_NAME].apply(clean_tweet).tolist()

# 3. Train Word2Vec
print("Training Word2Vec model...")
# Using the same parameters as before
model = Word2Vec(sentences=tokens_list, vector_size=100, window=5, min_count=5, workers=4)

words = list(model.wv.index_to_key)
vectors = model.wv[words]

print(f"Vocab size: {len(words)} unique words.")

# 4. Apply PCA
print("Applying PCA...")
pca = PCA(n_components=2)
result_pca = pca.fit_transform(vectors)

# 5. Evaluate Clusters (Loop 3 to 10)
print(f"Evaluating Cluster counts from {MIN_K} to {MAX_K}...")

silhouette_scores = []
db_scores = []
k_range = range(MIN_K, MAX_K + 1)

for k in k_range:
    # Initialize KMeans
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    
    # Fit on the PCA data
    labels = kmeans.fit_predict(result_pca)
    
    # Calculate Metrics
    # Silhouette: Higher is better (range -1 to 1)
    sil_score = silhouette_score(result_pca, labels)
    # Davies-Bouldin: Lower is better (0 to infinity)
    db_score = davies_bouldin_score(result_pca, labels)
    
    silhouette_scores.append(sil_score)
    db_scores.append(db_score)
    
    print(f"k={k}: Silhouette={sil_score:.4f}, Davies-Bouldin={db_score:.4f}")

# 6. Plotting the Evaluation Metrics
fig, ax1 = plt.subplots(figsize=(10, 6))

color = 'tab:blue'
ax1.set_xlabel('Number of Clusters (k)')
ax1.set_ylabel('Silhouette Score (Higher is better)', color=color)
ax1.plot(k_range, silhouette_scores, marker='o', color=color, linewidth=2, label='Silhouette')
ax1.tick_params(axis='y', labelcolor=color)

# Create a second y-axis for Davies-Bouldin
ax2 = ax1.twinx()  
color = 'tab:red'
ax2.set_ylabel('Davies-Bouldin Index (Lower is better)', color=color)
ax2.plot(k_range, db_scores, marker='s', linestyle='--', color=color, linewidth=2, label='Davies-Bouldin')
ax2.tick_params(axis='y', labelcolor=color)

plt.title('Clustering Evaluation: Silhouette vs Davies-Bouldin')
plt.grid(True, axis='x', linestyle='--')
plt.xticks(k_range)
plt.show()

# Optional: Suggest best k
best_sil_k = k_range[silhouette_scores.index(max(silhouette_scores))]
best_db_k = k_range[db_scores.index(min(db_scores))]

print("\n--- Recommendation ---")
print(f"Best k according to Silhouette Score (High): {best_sil_k}")
print(f"Best k according to Davies-Bouldin (Low): {best_db_k}")