from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def generate_text(model, tokenizer, prompt, max_length=50):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=max_length,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Load the fine-tuned model and tokenizer
model_path = "./prompt-tuned-model"  # Path where the model and tokenizer were saved
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)  # Ensure tokenizer is loaded from the same path
tokenizer.pad_token = tokenizer.eos_token

# Test prompts for inference
test_prompts = [
    "You as my friend will",
    "You as a film critic "
]

# Generate text using the fine-tuned model
print("Inference using the fine-tuned model:")
for prompt in test_prompts:
    generated_text = generate_text(model, tokenizer, prompt)
    print(f"\nPrompt: {prompt}")
    print(f"Generated: {generated_text}")
