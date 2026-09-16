# Text Similarity
from sentence_transformer import load_model
from sklearn.metrics.pairwise import cosine_similarity

model = load_model()


def text_similarity(sentence_pairs):
    # Encode every sentence in the pairs.
    # This returns the numpy
    pair_embeddings = model.encode(sentence_pairs)
    vectors = pair_embeddings

    # Calculate cosine similarity between the two vectors.

    '''
    E.g., Cosine Similarity gives the following resulting matrix:
                      Sentence 1    Sentence 2
    Sentence 1       1.0000        0.8
    Sentence 2       0.8           1.0000
    
    => We need to retrieve [0][1] for (sentence1, sentence2) = 0.8
    '''
    score = cosine_similarity(vectors)[0][1]
    sentence_a = sentence_pairs[0]
    sentence_b = sentence_pairs[1]
    print("\n")

    # Display the pair and its similarity score.
    print("-" * 60)
    print(f"Sentence A: {sentence_a}")
    print(f"Sentence B: {sentence_b}")
    print(f"Cosine similarity: {score:.4f}")
    print("-" * 60)


# Enter the sentence pairs
sentence1 = input("Enter sentence 1: ")
sentence2 = input("Enter sentence 2: ")
pairs = [sentence1, sentence2]
text_similarity(pairs)
