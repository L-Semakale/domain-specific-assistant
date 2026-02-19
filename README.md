# 🏥 Healthcare Assistant – Domain-Specific LLM Fine-Tuning

This project presents the development of a **domain-specific healthcare assistant** that was fine-tuned from a pre-trained Large Language Model (LLM) using parameter-efficient techniques. The model was customized to provide structured and medically relevant responses to healthcare-related queries.

The system was trained using LoRA (Low-Rank Adaptation) and deployed using Gradio on Hugging Face Spaces.

---

##  Project Overview

General-purpose language models often lack precision in specialized domains such as healthcare. This project demonstrates how a lightweight LLM can be adapted for domain-specific tasks through supervised fine-tuning.

The final system:

* Responds to healthcare-related questions
* Filters out non-medical queries
* Includes emergency keyword detection
* Provides responses in a structured format
* Is deployed for real-time interaction

---

##  Model Architecture

**Base Model:**
TinyLlama-1.1B-Chat-v1.0

**Fine-Tuning Method:**
LoRA (Low-Rank Adaptation) using the PEFT library

**Training Environment:**
Google Colab (Free GPU)

LoRA was used to reduce memory requirements by freezing base model weights and training only small rank-decomposition matrices in attention layers.

---

##  Dataset

**Dataset Used:**
`medalpaca/medical_meadow_medical_flashcards` (Hugging Face)

This dataset contains structured medical question–answer pairs derived from educational flashcards.

Preprocessing steps included:

* Removing incomplete entries
* Formatting into instruction–response templates
* Tokenization using the TinyLlama tokenizer
* Sequence truncation to 512 tokens
* Train/validation split (90/10)

Approximately 3,000 examples were used for training.

---

##  Fine-Tuning Configuration

| Parameter             | Value |
| --------------------- | ----- |
| Learning Rate         | 5e-5  |
| Batch Size            | 2     |
| Gradient Accumulation | 4     |
| Epochs                | 2     |
| Max Sequence Length   | 512   |
| Optimizer             | AdamW |

Multiple configurations were tested, and the final model was selected based on validation perplexity and qualitative performance.

---

##  Evaluation

The model was evaluated using both quantitative and qualitative methods:

* **Perplexity** (validation set)
* **ROUGE score**
* Base vs fine-tuned comparison
* Manual prompt testing

The fine-tuned model demonstrated:

* Lower perplexity
* Improved domain alignment
* More structured medical explanations
* Reduced off-topic generation

---

##  Deployment

The assistant is deployed using **Gradio** and hosted on **Hugging Face Spaces**.

Features include:

* Interactive chat interface
* Medical disclaimer
* Emergency keyword detection
* Out-of-domain query filtering

 **Live Demo:**
[Insert Hugging Face Space Link]

---

##  How to Run Locally

1. Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/domain-specific-assistant.git
cd domain-specific-assistant
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the application:

```bash
python app.py
```

The interface will launch locally.

---

##  Repository Structure

```
domain-specific-assistant/
│
├── app.py
├── healthcare_chatbot_finetuning.ipynb
├── requirements.txt
├── models/
│   └── lora_medassist/
│       ├── adapter_config.json
│       └── adapter_model.safetensors
├── experiment_results.csv
├── evaluation_results.json
└── README.md
```

##  Disclaimer

This chatbot is intended for educational and informational purposes only.
It is not a substitute for professional medical advice, diagnosis, or treatment.

Always consult a qualified healthcare provider regarding medical concerns.

---

##  License

This project is developed for academic purposes.

Tell me what style you prefer.
