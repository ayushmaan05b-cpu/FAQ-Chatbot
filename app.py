import nltk

nltk.download('punkt')
nltk.download('punkt_tab')

import streamlit as st
import nltk
import string

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')
nltk.download('stopwords')

faq_data = {
    "What is AI?": "AI stands for Artificial Intelligence.",
    "What is Python?": "Python is a programming language.",
    "What is NLP?": "NLP stands for Natural Language Processing.",
    "What is Machine Learning?": "Machine Learning is a subset of AI.",
    "What is Streamlit?": "Streamlit is used to build web apps.",
    "What is Data Science?": "Data Science is the study of data.",
    "What is Deep Learning?": "Deep Learning is a subset of Machine Learning.",
    "What is a Chatbot?": "A chatbot interacts with users automatically.",
    "What is Python used for?": "Python is used for web development, AI and data science.",
    "What is Cosine Similarity?": "It measures similarity between texts."
}

def preprocess(text):
    text = text.lower()
    tokens = word_tokenize(text)

    tokens = [
        word for word in tokens
        if word not in string.punctuation
    ]

    tokens = [
        word for word in tokens
        if word not in stopwords.words('english')
    ]

    return " ".join(tokens)

questions = list(faq_data.keys())

processed_questions = [
    preprocess(q)
    for q in questions
]

st.title("FAQ Chatbot")

user_question = st.text_input("Ask your question")

if user_question:

    processed_input = preprocess(user_question)

    vectorizer = TfidfVectorizer()

    vectors = vectorizer.fit_transform(
        processed_questions + [processed_input]
    )

    similarity = cosine_similarity(
        vectors[-1],
        vectors[:-1]
    )

    best_match = similarity.argmax()

    answer = faq_data[questions[best_match]]

    st.success(answer)