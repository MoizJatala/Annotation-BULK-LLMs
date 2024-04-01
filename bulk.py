import pandas as pd
from umap import UMAP
from sklearn.pipeline import make_pipeline 
# pip install "embetter[text]"
from embetter.text import SentenceEncoder
# Build a sentence encoder pipeline with UMAP at the end.
text_emb_pipeline = make_pipeline(
  SentenceEncoder('all-MiniLM-L6-v2'),
  UMAP()
)
# Load sentences
sentences = list(pd.read_csv("Corona_NLP_test.csv")['OriginalTweet'])
# Calculate embeddings 
X_tfm = text_emb_pipeline.fit_transform(sentences)
# Write to disk. Note! Text column must be named "text"
df = pd.DataFrame({"text": sentences})
df['x'] = X_tfm[:, 0]
df['y'] = X_tfm[:, 1]
df.to_csv("ready.csv")
df.to_csv("ready.csv", index=False)

