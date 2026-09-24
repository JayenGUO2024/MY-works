from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd
import torch

# 设置模型名称
model_name = "roberta-base"

# 创建样例数据
labeled_data = [
    {"Text": "The product is of great quality but expensive.", "Quality": 5, "Price": 2, "Delivery": -1},
    {"Text": "Fast delivery, but the product quality is mediocre.", "Quality": 3, "Price": -1, "Delivery": 5},
    {"Text": "Poor product quality, very dissatisfied.", "Quality": 1, "Price": -1, "Delivery": -1},
    {"Text": "The product is good, but I had to wait too long for delivery.", "Quality": 4, "Price": -1, "Delivery": 2},
    {"Text": "Quality could be improved, but overall a decent product.", "Quality": 3, "Price": -1, "Delivery": -1},
    {"Text": "Product is okay, but I wish it had a better price.", "Quality": 3, "Price": 2, "Delivery": -1},
]

# 转换为数据集
labeled_df = pd.DataFrame(labeled_data)
labeled_dataset = Dataset.from_pandas(labeled_df)

# 加载分词器
tokenizer = RobertaTokenizer.from_pretrained(model_name)

# 分词函数
def preprocess_function(examples):
    return tokenizer(examples["Text"], padding="max_length", truncation=True)

# 处理数据集
tokenized_dataset = labeled_dataset.map(preprocess_function, batched=True)

# 添加标签列
label_columns = ["Quality", "Price", "Delivery"]

def add_labels(example):
    example["labels"] = [float(example[label]) for label in label_columns]
    return example

tokenized_dataset = tokenized_dataset.map(add_labels)

# 数据集分割
train_test_split = tokenized_dataset.train_test_split(test_size=0.2)
train_dataset = train_test_split["train"]
test_dataset = train_test_split["test"]

# 加载模型
model = RobertaForSequenceClassification.from_pretrained(
    model_name,
    num_labels=3,  # 设置标签数量
    problem_type="multi_label_classification"
).to(torch.device("cpu"))  # 使用 CPU

# 修复的 TrainingArguments
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",  # 替换 evaluation_strategy
    save_strategy="epoch",  # 保存策略与评估策略一致
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir=None,
    logging_steps=10,
    load_best_model_at_end=True,
    use_cpu=True,  # 替换 no_cuda
)

# 自定义评估指标
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.sigmoid(torch.tensor(logits)).cpu().numpy()
    predictions = (predictions > 0.5).astype(int)

    accuracy_quality = (predictions[:, 0] == labels[:, 0]).mean()
    accuracy_price = (predictions[:, 1] == labels[:, 1]).mean()
    accuracy_delivery = (predictions[:, 2] == labels[:, 2]).mean()

    return {
        "accuracy_quality": accuracy_quality,
        "accuracy_price": accuracy_price,
        "accuracy_delivery": accuracy_delivery,
    }

# 初始化 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# 训练模型
trainer.train()

# 保存模型
model.save_pretrained("./roberta-review-model")
tokenizer.save_pretrained("./roberta-review-model")
