from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", add_bos_token=False, add_eos_token=False)

# Load the model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

# Encode input text
input_text = "Once upon a time"
inputs = tokenizer(input_text, return_tensors='pt')

# Generate text
outputs = model.generate(inputs.input_ids, max_length=50, num_return_sequences=1)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(generated_text)
