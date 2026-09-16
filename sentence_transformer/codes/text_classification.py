# Text classification
from sentence_transformer import load_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd

model = load_model()


def text_classification(classification_texts, classification_labels):
    # Convert the text examples into embedding features.
    classification_embeddings = model.encode(classification_texts)

    # Split the dataset into training and testing portions.
    X_train, X_test, y_train, y_test = train_test_split(
          classification_embeddings,
          classification_labels,
          test_size=0.30,
          random_state=42,
          stratify=classification_labels
      )

    # Create a Logistic Regression classifier.
    classifier = LogisticRegression(
          max_iter=1000,
          random_state=42
      )

    # Train the classifier using the embedding features.
    classifier.fit(X_train, y_train)

    # Predict labels for the test examples.
    y_pred = classifier.predict(X_test)

    # Evaluate the classifier.
    print("Accuracy:", f"{accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred, zero_division=0))


def main():
    # Load the classification dataset.
    classification_data = pd.read_csv('documents/classification_data.csv')

    # Separate the text examples from their labels.
    classification_texts = classification_data['text'].tolist()
    classification_labels = classification_data['label'].tolist()

    text_classification(classification_texts, classification_labels)


if __name__ == "__main__":
    main()
