import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax
import torch


# 1. Analyze a single review
def analyze_review(review_text, tokenizer, model):
    inputs = tokenizer(review_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    probs = softmax(outputs.logits, dim=-1).detach().cpu().numpy()[0]
    sentiment = ["negative", "neutral", "positive"][probs.argmax()]
    return sentiment


# 2. Analyze reviews from CSV
def analyze_csv(input_csv, tokenizer, model):
    df = pd.read_csv(input_csv)
    results = []

    for _, row in df.iterrows():
        category = row["Category"]
        review = row["Review"]
        overall_rating = row["Overall Rating"]

        # Predict sentiment using the model
        sentiment = analyze_review(review, tokenizer, model)
        results.append({
            "Category": category,
            "Overall Rating": overall_rating,
            "Review": review,
            "Sentiment": sentiment
        })

    return pd.DataFrame(results)


# 3. Visualize overall sentiment
def visualize_overall_sentiment(df):
    sentiment_counts = df["Sentiment"].value_counts()

    # Pie chart for sentiment distribution
    plt.figure(figsize=(8, 6))
    plt.pie(
        sentiment_counts,
        labels=sentiment_counts.index,
        autopct='%1.1f%%',
        colors=["red", "blue", "green"],
        startangle=140
    )
    plt.title("Overall Sentiment Distribution")
    plt.show()


# 4. Visualize sentiment by category
def visualize_sentiment_by_category(df):
    category_sentiment = df.groupby(["Category", "Sentiment"]).size().unstack(fill_value=0)

    # Bar chart for sentiment by category
    category_sentiment.plot(kind="bar", stacked=True, figsize=(10, 6), color=["red", "blue", "green"])
    plt.title("Sentiment Distribution by Category")
    plt.xlabel("Category")
    plt.ylabel("Number of Reviews")
    plt.legend(title="Sentiment")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Load model and tokenizer
    MODEL_PATH = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

    # Input CSV file
    input_csv = "E:/seiminar/DS2/Sentiment Analysis/JayenGUO2024/model_test/review_analysis_input.csv"

    # Analyze CSV
    results_df = analyze_csv(input_csv, tokenizer, model)

    # Save analyzed results to a new CSV
    results_df.to_csv("analyzed_results.csv", index=False)
    print("Analysis results saved to analyzed_results.csv!")

    # Visualize overall sentiment
    visualize_overall_sentiment(results_df)

    # Visualize sentiment by category
    visualize_sentiment_by_category(results_df)
