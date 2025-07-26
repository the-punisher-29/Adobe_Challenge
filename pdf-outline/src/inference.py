from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load model and tokenizer
model_id = "Qwen/Qwen2-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="cpu",
    torch_dtype=torch.float32
)

# Enable chat template if using Transformers >=4.38 and it's a chat model
tokenizer.use_default_system_prompt = False

# Define chat input
messages = [
    {"role": "user", "content": "Explain black holes in simple terms."}
]

# Format using tokenizer's chat template
input_ids = tokenizer.apply_chat_template(
    messages,
    return_tensors="pt"
).to(model.device)

# Generate response
with torch.no_grad():
    output = model.generate(
        input_ids=input_ids,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )

# Decode and print
response = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)
print("Qwen2:", response)
