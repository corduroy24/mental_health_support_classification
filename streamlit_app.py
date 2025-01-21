import streamlit as st
from transformers import pipeline
import pickle
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from sklearn.feature_extraction.text import TfidfVectorizer

from joblib import dump, load
import pandas as pd

# Show title and description.
st.title("Mental Health Support Smart Tags")
# st.write(
#     "This is a simple chatbot that uses OpenAI's GPT-3.5 model to generate responses. "
#     "To use this app, you need to provide an OpenAI API key, which you can get [here](https://platform.openai.com/account/api-keys). "
#     "You can also learn how to build this app step by step by [following our tutorial](https://docs.streamlit.io/develop/tutorials/llms/build-conversational-apps)."
# )

# Ask user for their OpenAI API key via `st.text_input`.
# Alternatively, you can store the API key in `./.streamlit/secrets.toml` and access it
# via `st.secrets`, see https://docs.streamlit.io/develop/concepts/connections/secrets-management

    # st.error()
with open('crisis_classifier.pkl', 'rb') as file:  
    model = pickle.load(file)

with open('tfidf_vectorizer.pkl', 'rb') as file:
    tfidf_vectorizer = pickle.load(file)

topic_model = BERTopic.load("my_model_6")
# bert_model = load("bert_model.joblib")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
topic_data = pd.read_csv("topic_data.csv")
message = st.text_area("What's on your mind ?")


if message.strip() != '':
    sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    result = sentiment_pipeline(message)[0]
    st.warning(result['label'])

    if result['label'] == "NEGATIVE": 
        # evaluate model 
        X_test_tfidf = tfidf_vectorizer.transform([message])
        y_predict = model.predict(X_test_tfidf)[0]
        y_prob = model.predict_proba(X_test_tfidf)[:, 1]
        st.write(y_prob)

        if y_predict == 1:
            st.error("⚠️ This message may indicate a crisis situation.")
            new_embeddings = embedding_model.encode([message])
            new_topics, new_probs = topic_model.transform([message], embeddings=new_embeddings)
            topic_name = str(topic_data[topic_data['Topic'] == new_topics[0]]['cleaned_llama'].iloc[0])
            st.info(topic_name)

        else:
            st.success("✅ This message does not indicate a crisis situation.")# st.info('Theme: ')    
