import streamlit as st
import requests

# Flask API base URL
API_URL = "http://127.0.0.1:5000"

st.title("Sentiment Analysis Frontend")

# Section: Upload Reviews
st.header("Upload Product Reviews")
product_id = st.text_input("Enter Product ID")

review_texts = st.text_area("Enter Reviews (One per line)")
if st.button("Upload Reviews"):
    if product_id and review_texts:
        reviews = [{"text": text.strip()} for text in review_texts.split("\n") if text.strip()]
        payload = {"product_id": product_id, "reviews": reviews}

        response = requests.post(f"{API_URL}/upload_reviews", json=payload)
        if response.status_code == 200:
            st.success("Reviews uploaded successfully!")
            st.json(response.json())
        else:
            st.error(f"Error: {response.json().get('error', 'Unknown error')}")
    else:
        st.warning("Please enter a Product ID and at least one review.")

# Section: Correct Analysis
st.header("Correct Sentiment Analysis")
review_id = st.text_input("Enter Review ID for Correction")
correction_type = st.selectbox("Select Correction Type", [
    "missing_quality", "missing_delivery", "missing_price", "missing_packaging",
    "missing_service", "incorrect_sentiment", "missing_category", "unclear_analysis",
    "misclassified_text", "text_not_extracted", "too_general", "neutral_sentiment_check",
    "positive_but_negative_context", "more_context_needed"
])

if st.button("Correct Analysis"):
    if review_id and correction_type:
        payload = {"review_id": review_id, "correction_type": correction_type}
        response = requests.post(f"{API_URL}/correct_analysis", json=payload)

        if response.status_code == 200:
            st.success("Sentiment analysis corrected!")
            st.json(response.json())
        else:
            st.error(f"Error: {response.json().get('error', 'Unknown error')}")
    else:
        st.warning("Please enter a Review ID and select a correction type.")
