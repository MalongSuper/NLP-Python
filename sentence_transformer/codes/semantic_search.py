# Semantic Search
from sentence_transformer import load_model
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

model = load_model()


def semantic_search(query, documents):
    # Encode the documents and the query.
    document_embeddings = model.encode(documents)
    query_embedding = model.encode([query])

    # Calculate the similarity between the query and every document.
    similarity_scores = cosine_similarity(
          query_embedding,
          document_embeddings
      )[0]

    # Rank document indexes from the highest score to the lowest score.
    ranked_indexes = np.argsort(similarity_scores)[::-1]

    print("\n")
    print("-" * 100)
    # Display the ranked search results -> Get the top 10 match.
    for rank, document_index in enumerate(ranked_indexes[:10], start=1):
        print(f"{rank}. Score: {similarity_scores[document_index]:.4f}")
        print(f"   {documents[document_index]}")

    print("-" * 100)


def read_documents(path):
    with open(f'documents/{path}', 'r', encoding='utf-8') as file:
        documents = [line.strip() for line in file if line.strip()]
    return documents


def main():
    documents = read_documents('documents.txt')
    query = input("Enter query: ")
    semantic_search(query, documents)


if __name__ == '__main__':
    main()
