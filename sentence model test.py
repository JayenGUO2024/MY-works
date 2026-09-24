from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load the fine-tuned model and tokenizer
model_path = "./tweets_and_semeval_model"  # Replace with your model directory
model = GPT2LMHeadModel.from_pretrained(model_path)
tokenizer = GPT2Tokenizer.from_pretrained(model_path)

# 扩展情感关键词
positive_keywords = ["good", "excellent", "amazing", "great", "delicious", "impressive", "remarkable", "satisfactory"]
negative_keywords = ["bad", "terrible", "unhelpful", "frustrating", "disappointing", "crashes", "poor", "unsatisfactory", "inattentive"]
neutral_keywords = ["ok", "fine", "average", "acceptable", "decent"]

def analyze_sentiment(text):
    """
    Analyze the sentiment of a given text using the fine-tuned GPT-2 model.

    Args:
        text (str): The input text to analyze.

    Returns:
        str: The predicted sentiment or aspects and sentiments.
    """
    # Encode the input text
    input_ids = tokenizer.encode(text, return_tensors="pt")
    attention_mask = torch.ones_like(input_ids)  # Construct attention mask

    # Add an EOS token to signal end of input if necessary
    input_ids = torch.cat([input_ids, torch.tensor([[tokenizer.eos_token_id]])], dim=1)
    attention_mask = torch.ones_like(input_ids)

    # Generate predictions using the model
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,  # Add attention mask
            max_new_tokens=50,  # Limit the number of generated tokens
            num_beams=10,   # Beam search for better predictions
            no_repeat_ngram_size=2,  # Avoid repeated phrases
            early_stopping=True,
            temperature=0.7,  # Adjust randomness of generation
            pad_token_id=tokenizer.eos_token_id  # Ensure proper padding behavior
        )

    # Decode the generated IDs to text
    predicted_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f"Raw Output: {predicted_text}")  # Print raw output for debugging

    # 优化情感判断逻辑
    sentiment = None
    if "positive" in predicted_text.lower():
        sentiment = "Positive"
    elif "negative" in predicted_text.lower():
        sentiment = "Negative"
    elif "neutral" in predicted_text.lower():
        sentiment = "Neutral"

    # 如果模型未明确给出情感，使用关键词判断
    if not sentiment:
        positive_count = sum(word in predicted_text.lower() for word in positive_keywords)
        negative_count = sum(word in predicted_text.lower() for word in negative_keywords)
        neutral_count = sum(word in predicted_text.lower() for word in neutral_keywords)

        # 综合判断情感
        if positive_count > negative_count and positive_count > neutral_count:
            sentiment = "Positive"
        elif negative_count > positive_count and negative_count > neutral_count:
            sentiment = "Negative"
        elif neutral_count > positive_count and neutral_count > negative_count:
            sentiment = "Neutral"
        else:
            sentiment = "Uncertain"

    # 转折词分析
    if "but" in text.lower() or "however" in text.lower() or "although" in text.lower():
        if sentiment == "Positive" and any(word in text.lower() for word in negative_keywords):
            sentiment = "Mixed - Positive with Negative aspects"
        elif sentiment == "Negative" and any(word in text.lower() for word in positive_keywords):
            sentiment = "Mixed - Negative with Positive aspects"

    return sentiment

# Test with complex sentences
test_texts = [
    "Although the delivery was significantly delayed, which initially frustrated me, the product ultimately surpassed my expectations with its outstanding craftsmanship and remarkable build quality.",
    "Despite multiple attempts to seek assistance, the customer support team remained unresponsive and the instructions provided were vague at best, turning the setup process into an unnecessarily challenging and frustrating experience.",
    "While I was thoroughly impressed by the elegant design and exceptional performance of the product, the disappointingly short battery life significantly detracted from what could have been a flawless premium offering.",
    "Even though the app boasts a sleek and user-friendly interface that initially made a great impression, its frequent crashes and persistent bugs have rendered it nearly unusable for any serious tasks.",
    "The restaurant offered an exquisite ambiance complemented by mouthwatering dishes; however, the extremely sluggish and inattentive service dampened what could have been a truly exceptional dining experience."
]

# Analyze sentiment for each test text
for text in test_texts:
    sentiment = analyze_sentiment(text)
    print(f"Input: {text}")
    print(f"Predicted Sentiment: {sentiment}")
    print("-" * 50)

