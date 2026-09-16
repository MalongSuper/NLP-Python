# Paraphrase Detection
from sentence_transformer import load_model
from sklearn.metrics.pairwise import cosine_similarity


model = load_model()


def detect_paraphrase(sentence_a, sentence_b, threshold=0.70):
    # Encode both sentences as vectors.
    embeddings = model.encode([sentence_a, sentence_b])

    # Calculate cosine similarity between the two sentence vectors.
    score = cosine_similarity(
        embeddings[0].reshape(1, -1),
        embeddings[1].reshape(1, -1)
    )[0][0]

    # Decide whether the pair is likely to be a paraphrase.
    is_paraphrase = score >= threshold

    # Return both the score and the decision.
    return score, is_paraphrase


# Enter the sentence pairs
sentence1 = input("Enter sentence 1: ")
sentence2 = input("Enter sentence 2: ")
score, decision = detect_paraphrase(sentence1, sentence2)

print("\n")
print("-" * 60)
print(f"Sentence A: {sentence1}")
print(f"Sentence B: {sentence2}")
print(f"Similarity: {score:.4f}")
print(f"Likely paraphrase: {decision}")
print("-" * 60)
