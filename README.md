# Spam Email Classifier using NLP

This project builds a machine learning model that classifies SMS messages as **Spam or Ham (Not Spam)** using Natural Language Processing techniques.

The model is trained on the **SMS Spam Collection dataset** containing over 5,500 labeled messages.

## Dataset

SMS Spam Collection Dataset  
Source: UCI Machine Learning Repository

The dataset contains:

- 5,572 SMS messages
- Labelled as spam or ham

## Technologies Used

- Python
- Pandas
- scikit-learn
- TF-IDF Vectorization
- Multinomial Naive Bayes
- Matplotlib

## Machine Learning Pipeline

1. Load dataset
2. Clean and preprocess text
3. Convert text to numerical features using TF-IDF
4. Train a Naive Bayes classifier
5. Evaluate model performance

## Model Performance

Typical accuracy:

98–99%

## Example Prediction

Input message:

"Congratulations! You have won a free iPhone. Click here to claim."

Prediction:

Spam

## Skills Demonstrated

- Natural Language Processing
- Text Feature Engineering
- Machine Learning Classification
- Model Evaluation
- Data Visualization

## Future Improvements

- Use deep learning models
- Add text preprocessing (stemming, lemmatization)
- Deploy as a web app
