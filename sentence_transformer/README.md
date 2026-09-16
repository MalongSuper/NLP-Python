# Sentence Transformer

This repository introduces `sentence-transformers/all-MiniLM-L6-v2`, a pretrained English sentence-embedding model that converts sentences and short paragraphs into 384-dimensional dense vectors. These embeddings capture semantic meaning, allowing texts with similar meanings to have similar vector representations.

## How It Works

The model uses a MiniLM Transformer encoder followed by mean pooling to produce a fixed-size sentence embedding. It was fine-tuned using a contrastive learning objective on over one billion sentence pairs, helping it learn meaningful relationships between texts.

## Import and Usage

Install the library and load the pretrained model:

```python
!pip install -U sentence-transformers

from sentence_transformers import SentenceTransformer

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

sentences = [
    'Machine learning is a branch of AI.',
    'Artificial intelligence includes machine learning.'
]

embeddings = model.encode(sentences)

print(embeddings.shape)
# (2, 384)
```

## Use Cases

* **Text Classification:** Use embeddings as features for models such as Logistic Regression.
* **Similarity:** Compare the semantic meaning of sentences using cosine similarity.
* **FAQ:** Match user questions with relevant predefined answers.
* **Recommender Systems:** Recommend documents, products, or content based on semantic similarity.
* **Semantic Search:** Retrieve relevant information based on meaning rather than exact keyword matches.

For a detailed model description, architecture, training procedure, and intended uses, visit the [Hugging Face model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).
