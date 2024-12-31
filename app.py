from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import torch
import os
from datasets import load_dataset

# Model and tokenizer setup
model_name = "bigscience/bloomz-560m"
NUM_EPOCHS = 6
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

# Function to generate outputs
def get_outputs(model, inputs, max_new_tokens=100):
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=max_new_tokens,
        repetition_penalty=1.5,
        early_stopping=True,
        eos_token_id=tokenizer.eos_token_id,
    )
    return outputs

# Test prompts
test_prompts = [
    "I want you to act as a motivational coach.",
    "There are two nice things that should matter to you:"
]

# Generate text before training
print("Before training:")
before_training_results = {}
for prompt in test_prompts:
    input_prompt = tokenizer(prompt, return_tensors="pt")
    outputs = get_outputs(model, input_prompt, max_new_tokens=50)
    generated = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    before_training_results[prompt] = generated
    print(f"\nPrompt: {prompt}")
    print(f"Generated: {generated}")

# Load dataset
dataset_prompt = "fka/awesome-chatgpt-prompts"
data_prompt = load_dataset(dataset_prompt)
data_prompt = data_prompt.map(lambda samples: tokenizer(samples["prompt"], padding=True, truncation=True, max_length=512), batched=True)
train_sample_prompt = data_prompt["train"].select(range(50))

# Extract necessary fields from train_sample_prompt
train_encodings = {
    "input_ids": [sample["input_ids"] for sample in train_sample_prompt],
    "attention_mask": [sample["attention_mask"] for sample in train_sample_prompt]
}

# Define a simple dataset
class PromptDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

# Create dataset and dataloader
dataset = PromptDataset(train_encodings)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# Training settings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
num_epochs = 1

# Training loop
model.train()
for epoch in range(num_epochs):
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch, labels=batch['input_ids'])
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(f"Loss: {loss.item():.4f}")

# Save the model
model.save_pretrained("./prompt-tuned-model")
tokenizer.save_pretrained("./prompt-tuned-model")

# Generate text after training
print("\nAfter training:")
after_training_results = {}
for prompt in test_prompts:
    input_prompt = tokenizer(prompt, return_tensors="pt")
    outputs = get_outputs(model, input_prompt, max_new_tokens=50)
    generated = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    after_training_results[prompt] = generated
    print(f"\nPrompt: {prompt}")
    print(f"Generated: {generated}")

# Compare results
print("\nComparison of results before and after training:")
for prompt in test_prompts:
    print(f"\nPrompt: {prompt}")
    print(f"Before training: {before_training_results[prompt]}")
    print(f"After training: {after_training_results[prompt]}")
