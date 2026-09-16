# Recommender System for Courses
from sentence_transformer import load_model
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

model = load_model()


def recommend_system(documents, descriptions, user_query, title, content):
    # Encode the user's query.
    query_embedding = model.encode([user_query])

    # Encode all descriptions.
    descriptions_embeddings = model.encode(descriptions)

    # Calculate similarity between the query and every technical document.
    scores = cosine_similarity(
          query_embedding,
          descriptions_embeddings
      )[0]

    # Rank documents by similarity score.
    ranking = np.argsort(scores)[::-1]

    # Display recommended technical documents.
    print("-" * 60)
    print(f"Recommendations for: '{user_query}'\n")
    # Select top 10 best matches
    for rank, doc_index in enumerate(ranking[:10], start=1):
        print(f"{rank}. {documents[doc_index][title]}")
        print(f"   Score: {scores[doc_index]:.4f}")
        print(f"   {documents[doc_index][content]}")
        print("-" * 60)


def main():
    courses = []

    with open('documents/courses.txt', 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            # Split only at the first " — ".
            # The 1 means split only once:
            title, description = line.split(' — ', 1)

            courses.append({
                'title': title,
                'description': description
            })

    courses_descriptions = [course["description"] for course in courses]
    learner_interest = input("Learner's interest: ")
    recommend_system(courses, courses_descriptions, learner_interest,
                     title='title', content='description')


if __name__ == '__main__':
    main()
