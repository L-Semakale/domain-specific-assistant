import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# =============================
# Model Configuration
# =============================

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ADAPTER_PATH = "models/lora_medassist"

print("Loading model...")

# Detect device automatically
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None
)

# Load tokenizer from BASE model (important)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

# Attach LoRA adapter from LOCAL folder
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

model.eval()

print("Model loaded successfully!")

# =============================
# Text Generation
# =============================

def generate_response(prompt, max_new_tokens=256):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# =============================
# Chat Function
# =============================

def chat_with_healthcare_bot(message, history):

    emergency_keywords = [
        "emergency", "urgent", "dying",
        "suicide", "chest pain", "heart attack", "stroke"
    ]

    if any(word in message.lower() for word in emergency_keywords):
        return (
            "🚨 This appears to be a medical emergency. "
            "Please contact your local emergency services immediately "
            "(e.g., 911 in the US, 112 in many countries)."
        )

    non_medical_keywords = [
        "joke", "movie", "football",
        "bitcoin", "crypto", "politics"
    ]

    if any(word in message.lower() for word in non_medical_keywords):
        return "I am a healthcare-focused assistant. Please ask a medical-related question."

    prompt = f"""### Instruction:
Provide a clear, accurate, and professional answer to the following medical question.

### Question:
{message}

### Response:
"""

    response = generate_response(prompt)

    if "### Response:" in response:
        response = response.split("### Response:")[1].strip()

    return response

# =============================
# Gradio Interface
# =============================

disclaimer = (
    "⚠️ MEDICAL DISCLAIMER:\n"
    "This chatbot is for educational purposes only and is not a substitute "
    "for professional medical advice. Always consult a qualified healthcare provider."
)

demo = gr.ChatInterface(
    fn=chat_with_healthcare_bot,
    title="Healthcare Assistant Chatbot",
    description=disclaimer,
    theme="soft"
)

if __name__ == "__main__":
    demo.launch()
