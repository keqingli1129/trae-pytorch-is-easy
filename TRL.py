from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import torch
import evaluate
from transformers import DataCollatorWithPadding, pipeline



dataset = load_dataset("csv", data_files="./ChnSentiCorp_htl_all.csv", split="train")
dataset = dataset.filter(lambda x: x["review"] is not None)
datasets= dataset.train_test_split(test_size=0.2)
print(datasets)

tokenizer = AutoTokenizer.from_pretrained("hfl/rbt3")

def process_func(examples):
    tokenized_examples = tokenizer(examples["review"], max_length=128, padding="max_length", truncation=True)
    tokenized_examples["labels"] = examples["label"]
    return tokenized_examples
    

tokenized_datasets = datasets.map(process_func, batched=True, remove_columns=datasets["train"].column_names)
print(tokenized_datasets)
print(tokenized_datasets["train"][0])

model = AutoModelForSequenceClassification.from_pretrained("hfl/rbt3")
print(model.config)

acc_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

def eval_metric(eval_predict):
    predictions = eval_predict.predictions
    labels = eval_predict.label_ids
    predictions = predictions.argmax(axis=-1)
    acc = acc_metric.compute(predictions=predictions, references=labels)
    f1 = f1_metric.compute(predictions=predictions, references=labels)
    acc.update(f1)
    return acc

train_args = TrainingArguments(output_dir="./checkpoints",      # 输出文件夹
                               per_device_train_batch_size=64,  # 训练时的batch_size
                               per_device_eval_batch_size=128,  # 验证时的batch_size
                               logging_steps=10,                # log 打印的频率
                               eval_strategy="epoch",     # 评估策略
                               save_strategy="epoch",           # 保存策略
                               save_total_limit=3,              # 最大保存数
                               learning_rate=2e-5,              # 学习率
                               weight_decay=0.01,               # weight_decay
                               metric_for_best_model="f1",      # 设定评估指标
                               load_best_model_at_end=True)     # 训练完成后加载最优模型

trainer = Trainer(model=model, 
                  args=train_args, 
                  train_dataset=tokenized_datasets["train"], 
                  eval_dataset=tokenized_datasets["test"], 
                  data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
                  compute_metrics=eval_metric)

trainer.train()
trainer.evaluate(tokenized_datasets["test"])
trainer.predict(tokenized_datasets["test"])

id2_label = {0: "negative", 1: "positive"}
model.config.id2label = id2_label
pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)
print(pipe("我不喜欢这个商品"))
print(pipe("这个商品很赞")) 