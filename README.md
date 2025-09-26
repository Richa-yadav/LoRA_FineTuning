🏴‍☠️ Fine-Tuning LLaMA 3.2 1B on Pirate Ultrachat

This repository contains code and instructions for fine-tuning the LLaMA 3.2 1B Instruct model using the Pirate Ultrachat dataset
.
The goal is to teach the model to speak like a pirate while preserving its base reasoning abilities.

📌 Objectives

Fine-tune LLaMA 3.2 1B with LoRA adapters on a subset of 500 training examples (due to Colab storage limits).

Track training loss with a custom callback and visualize the loss curve.

Save the fine-tuned model locally.

Run inference and compare Base vs Fine-tuned outputs on the same prompts.

⚙️ Setup
# Clone repo
git clone https://github.com/Richa-yadav/LoRA_FineTuning.git

cd llama-pirate-finetune

# Install dependencies
pip install -r requirements.txt


requirements.txt should include:

transformers
datasets
peft
bitsandbytes
accelerate
sentencepiece
torch

📂 Dataset

We use a small subset of Pirate Ultrachat (500 samples):

from datasets import load_dataset

dataset = load_dataset("winglian/pirate-ultrachat-10k")
train_dataset = dataset["train"].select(range(500))


Each example contains a short conversation with a pirate-themed assistant.

🏋️ Training

Training is done with LoRA adapters for efficiency.
We log loss values during training and plot them after completion.

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator,
    callbacks=[LossTracker()]
)

trainer.train()


Loss is tracked and visualized with matplotlib.

📊 Results
Training Loss Curve

The loss decreases smoothly, confirming successful fine-tuning.

🧪 Inference

We compare Base vs Fine-tuned models using the same prompts.

Example:

Prompt:

Describe the sea at night in your own words.


Base Model:

The sea at night is calm and quiet, reflecting the moonlight with gentle waves.


Fine-tuned Pirate Model:

Arrr! The sea be a dark mistress at night, glimmerin’ with moonfire on her restless waves, whisperin’ secrets o’ the deep.

🔬 Test Prompts

We evaluate with both pirate-friendly prompts and general knowledge prompts.

Pirate Prompts

Describe the sea at night in your own words.

What should I do if I find a hidden treasure chest?

Write me a pirate song about rum and adventure.

Give me pirate-style instructions for cooking fish.

Explain the meaning of loyalty among a pirate crew.

General Prompts

What is the capital of France?

Explain photosynthesis in simple terms.

What’s the difference between a comet and an asteroid?

Can you write a short poem about friendship?

What is machine learning?

📌 Key Learnings

Small-scale fine-tuning (500 samples) works in Colab without Pro.

LoRA adapters allow efficient training with limited VRAM.

Base model responses are factual, while fine-tuned responses adopt pirate style.

🚀 Future Work

Train on larger subsets for richer pirate vocabulary.

Evaluate using BLEU/ROUGE for stylistic alignment.

Try multi-turn conversations instead of single-turn prompts.

👩‍💻 Author

Richa Kumari
