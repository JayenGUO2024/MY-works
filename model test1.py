import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax
import torch
import matplotlib.pyplot as plt


def analyze_review(review_text, tokenizer, model):
    # Tokenize and encode the review
    inputs = tokenizer(review_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    probs = softmax(outputs.logits, dim=-1).detach().cpu().numpy()[0]
    sentiment = ["negative", "neutral", "positive"][probs.argmax()]  # Map sentiment to labels
    return sentiment, probs


def factor_analysis(review_text):
    # A keyword-based factor analysis
    factors = {"Quality": 0, "Price": 0, "Delivery": 0}
    if "quality" in review_text.lower():
        factors["Quality"] = 1
    if "price" in review_text.lower():
        factors["Price"] = 1
    if "delivery" in review_text.lower() or "shipping" in review_text.lower():
        factors["Delivery"] = 1
    return factors


def analyze_and_predict(input_csv, tokenizer, model):
    df = pd.read_csv(input_csv)
    results = []

    for _, row in df.iterrows():
        category = row["Category"]
        review = row["Review"]
        overall_rating = row["Overall Rating"]

        # Analyze review sentiment
        sentiment, sentiment_probs = analyze_review(review, tokenizer, model)

        # Analyze factors contributing to the sentiment
        factors = factor_analysis(review)

        # Interpret factors contributing to positive/negative sentiment
        factor_contributions = {
            "Positive Factors": [],
            "Negative Factors": []
        }

        if sentiment == "positive":
            for factor, mentioned in factors.items():
                if mentioned:
                    factor_contributions["Positive Factors"].append(factor)
        elif sentiment == "negative":
            for factor, mentioned in factors.items():
                if mentioned:
                    factor_contributions["Negative Factors"].append(factor)

        # Append results
        results.append({
            "Category": category,
            "Overall Rating": overall_rating,
            "Review": review,
            "Sentiment": sentiment,
            "Sentiment Probabilities": sentiment_probs,
            "Positive Factors": ", ".join(factor_contributions["Positive Factors"]),
            "Negative Factors": ", ".join(factor_contributions["Negative Factors"]),
        })

    return pd.DataFrame(results)


def visualize_results(df):
    # Initialize counts for each factor and sentiment
    factor_counts = {
        "Quality": {"positive": 0, "negative": 0, "neutral": 0},
        "Price": {"positive": 0, "negative": 0, "neutral": 0},
        "Delivery": {"positive": 0, "negative": 0, "neutral": 0}
    }

    # Count mentions of factors by sentiment
    for _, row in df.iterrows():
        sentiment = row["Sentiment"]
        for factor in ["Quality", "Price", "Delivery"]:
            if factor in row["Positive Factors"]:
                factor_counts[factor]["positive"] += 1
            if factor in row["Negative Factors"]:
                factor_counts[factor]["negative"] += 1

    # Prepare data for plotting
    factors = ["Quality", "Price", "Delivery"]
    positive_counts = [factor_counts[factor]["positive"] for factor in factors]
    negative_counts = [factor_counts[factor]["negative"] for factor in factors]

    # Plot bar chart for factor contributions
    x = range(len(factors))
    width = 0.4

    plt.figure(figsize=(10, 6))
    plt.bar(x, positive_counts, width, label="Positive Mentions", alpha=0.7, color="green")
    plt.bar([i + width for i in x], negative_counts, width, label="Negative Mentions", alpha=0.7, color="red")

    plt.xticks([i + width / 2 for i in x], factors)
    plt.xlabel("Factors")
    plt.ylabel("Mentions Count")
    plt.title("Factor Contributions to Sentiment")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    MODEL_PATH = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

    input_csv = "E:/seiminar/DS2/Sentiment Analysis/JayenGUO2024/model_test/review_analysis_input.csv"
    results_df = analyze_and_predict(input_csv, tokenizer, model)

    # Display a portion of the results for verification
    print(results_df.head())

    # Save the results to a new CSV file for analysis
    results_df.to_csv("E:/seiminar/DS2/Sentiment Analysis/JayenGUO2024/model_test/predicted_results.csv", index=False)

    # Visualize results
    visualize_results(results_df)

