from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import re

# 加载模型和分词器
model_path = "./tweets_and_semeval_model"  # 替换为你的模型路径
model = GPT2LMHeadModel.from_pretrained(model_path)
tokenizer = GPT2Tokenizer.from_pretrained(model_path)

# 情感关键词
positive_keywords = ["good", "excellent", "amazing", "great", "delicious", "impressive", "remarkable", "satisfactory"]
negative_keywords = ["bad", "terrible", "unhelpful", "frustrating", "disappointing", "crashes", "poor", "unsatisfactory"]
neutral_keywords = ["ok", "fine", "average", "acceptable", "decent"]

# 分割规则：保持子句语义完整
def split_sentences(text):
    """
    基于标点符号和连接词分割句子，同时避免过短短语的孤立。
    """
    sentences = re.split(r'[.!?]', text)
    refined_sentences = []
    for sentence in sentences:
        if ' but ' in sentence or ' and ' in sentence or ' or ' in sentence:
            refined_sentences.extend(re.split(r' but | and | or ', sentence))
        else:
            refined_sentences.append(sentence.strip())
    # 合并过短短语到前一个句子
    merged_sentences = []
    for sent in refined_sentences:
        if len(sent.split()) < 3 and merged_sentences:
            merged_sentences[-1] += f" {sent}"
        else:
            merged_sentences.append(sent.strip())
    return [s for s in merged_sentences if s]

# 情感分析函数
def analyze_sentiment(text):
    """
    对输入文本进行情感分析。
    """
    input_ids = tokenizer.encode(text, return_tensors="pt")
    input_ids = torch.cat([input_ids, torch.tensor([[tokenizer.eos_token_id]])], dim=1).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=50,
            num_beams=5,
            no_repeat_ngram_size=2,
            early_stopping=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    predicted_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # 使用关键词匹配情感
    if any(word in predicted_text.lower() for word in positive_keywords):
        return "Positive"
    elif any(word in predicted_text.lower() for word in negative_keywords):
        return "Negative"
    elif any(word in predicted_text.lower() for word in neutral_keywords):
        return "Neutral"
    else:
        return "Uncertain"

# 分割并分类句子
def split_and_classify(text):
    """
    对句子进行分割并分类情感。
    """
    sentences = split_sentences(text)
    classified_sentences = {"Positive": [], "Negative": [], "Neutral": [], "Uncertain": []}

    for sentence in sentences:
        sentiment = analyze_sentiment(sentence)
        classified_sentences[sentiment].append(sentence)

    return classified_sentences

# 示例输入
input_text = "Despite multiple attempts to seek assistance, the customer support team remained unresponsive and the instructions provided were vague at best, turning the setup process into an unnecessarily challenging and frustrating experience"

# 分割并分类
classified_result = split_and_classify(input_text)

# 输出分类结果
for sentiment, sentences in classified_result.items():
    print(f"{sentiment}:")
    for sent in sentences:
        print(f"  - {sent}")
