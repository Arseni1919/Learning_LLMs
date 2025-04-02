from transformers import AutoModelForSequenceClassification, get_scheduler
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
import evaluate

from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch
from tqdm import tqdm


def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

# general parameters
checkpoint = "bert-base-uncased"
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'You are using the {device} device.')

# prepare the data
raw_datasets = load_dataset("glue", "mrpc")
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(["sentence1", "sentence2", "idx"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")
train_dataloader = DataLoader(
    tokenized_datasets["train"], shuffle=True, batch_size=8, collate_fn=data_collator
)
eval_dataloader = DataLoader(
    tokenized_datasets["validation"], batch_size=8, collate_fn=data_collator
)

"""
# To quickly check the dataloader, run this:
for batch in train_dataloader:
    break
{k: v.shape for k, v in batch.items()}  # check shapes
"""

# prepare the model for training
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
optimizer = AdamW(model.parameters(), lr=3e-5)
model.to(device)
num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

# prepare the model for evaluation
metric = evaluate.load("glue", "mrpc")

progress_bar = tqdm(range(num_training_steps))

for epoch in range(num_epochs):

    model.train()
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

    model.eval()
    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])

    print(metric.compute())






