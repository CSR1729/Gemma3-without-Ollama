from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

path = r"C:\Users\Chandan Sagar Rana\Desktop\Gemma\gemma-3-270m"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    path,
    local_files_only=True
)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    path,
    local_files_only=True,
    device_map="auto",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)

model.eval()

prompt = (
    "Write a full-length, vivid story about the jungle king lion. "
    "The story should have a clear beginning, conflict, and resolution, "
    "with rich descriptions and emotions.\n\nStory:\n"
)

# Tokenize
inputs = tokenizer(
    prompt,
    return_tensors="pt"
)

inputs = {k: v.to(model.device) for k, v in inputs.items()}

# Generate
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=500,          # allow a full story
        temperature=0.8,             # creativity
        top_p=0.95,                  # nucleus sampling
        do_sample=True,              # IMPORTANT for storytelling
        repetition_penalty=1.1,      # reduce looping
        eos_token_id=tokenizer.eos_token_id
    )

# Decode
story = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(story)
