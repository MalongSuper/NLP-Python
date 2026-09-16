# Retrieval-Based FAQ Chatbot
from sentence_transformer import load_model
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

model = load_model()


def answer_faq(user_question, threshold=0.45):
    # Encode the incoming user question.
    question_embedding = model.encode([user_question])

    # Calculate similarity between the user question and every FAQ question.
    scores = cosine_similarity(
        question_embedding,
        faq_embeddings
    )[0]

    # Find the index of the most similar FAQ question.
    best_index = int(np.argmax(scores))

    # Retrieve the highest similarity score.
    best_score = float(scores[best_index])

    # Reject the question when no FAQ is sufficiently similar.
    if best_score < threshold:
        return {
            "answer": "Sorry, I could not find a relevant FAQ answer.",
            "matched_question": None,
            "score": best_score
        }

    # Return the answer associated with the best matching FAQ.
    return {
        "answer": faq_data[best_index]["answer"],
        "matched_question": faq_data[best_index]["question"],
        "score": best_score
    }


# Retrieve the FAQ data
faq_data = []

with open('documents/faq_data.txt', 'r', encoding='utf-8') as file:
    content = file.read().strip()

# Separate each FAQ using the blank lines.
entries = content.split('\n\n')

for entry in entries:
    lines = entry.splitlines()

    question = lines[0].replace('Question: ', '', 1)
    answer = lines[1].replace('Answer: ', '', 1)

    faq_data.append({
        'question': question,
        'answer': answer
    })


# Extract FAQ questions from the knowledge base.
faq_questions = [item['question'] for item in faq_data]

# Encode all FAQ questions once so they can be reused.
faq_embeddings = model.encode(faq_questions)


result = answer_faq(input("Enter your question: "), threshold=0.45)

print("\n")

print("-" * 80)
print("Matched FAQ:", result["matched_question"])
print("Similarity score:", f"{result['score']:.4f}")
print("Chatbot answer:", result["answer"])
print("-" * 80)
