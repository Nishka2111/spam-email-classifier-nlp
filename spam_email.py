import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
df = pd.read_csv(
    "SMSSpamCollection",
    sep="\t",
    names=["label", "text"]
)

# Convert labels to numbers
df["label"] = df["label"].map({"ham":0, "spam":1})

print("Dataset shape:", df.shape)
print(df.head())

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    df["text"],
    df["label"],
    test_size=0.2,
    random_state=42
)

# Convert text to TF-IDF vectors
vectorizer = TfidfVectorizer(stop_words="english")

X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)

# Train model
model = MultinomialNB()
model.fit(X_train_vectors, y_train)

# Predictions
predictions = model.predict(X_test_vectors)

# Accuracy
accuracy = accuracy_score(y_test, predictions)
print("\nModel Accuracy:", accuracy)

# Classification report
print("\nClassification Report:\n")
print(classification_report(y_test, predictions))

# Confusion Matrix
cm = confusion_matrix(y_test, predictions)

print("\nConfusion Matrix:")
print(cm)

# Plot confusion matrix
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Test custom message
email = input("\nEnter a message to classify:\n> ")

email_vector = vectorizer.transform([email])

prediction = model.predict(email_vector)

if prediction[0] == 1:
    print("\nPrediction: SPAM")
else:
    print("\nPrediction: NOT SPAM")