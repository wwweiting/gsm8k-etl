import json
import os
import numpy as np
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from sklearn.cluster import KMeans

# 1. Setup
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 2. Load Data
input_file = 'data/structured_dataset_completeQ.json'
with open(input_file, 'r', encoding='utf-8') as f:
    # ★ 注意：這裡只取前 10 筆，若要跑全部請拿掉 [:10]
    data = json.load(f)[:10]

questions = [item['query']['question'] for item in data]
ids = [item['context']['dataset_id'] for item in data]

print(f"Loaded {len(questions)} questions for clustering...")

# 3. Get Embeddings
def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding

print("Generating embeddings...")
matrix = np.array([get_embedding(q) for q in questions])

# 4. K-Means Clustering
n_clusters = 7
kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
kmeans.fit(matrix)
labels = kmeans.labels_

# 5. Save to CSV
output_filename = "method1_clustering_results.csv"

# 建立 DataFrame
df = pd.DataFrame({
    "question_id": ids,
    "categories": [f"Cluster {label}" for label in labels]
})

df.to_csv(output_filename, index=False, encoding='utf-8-sig')
print(f"Done! Results saved to {output_filename}")
print(df)