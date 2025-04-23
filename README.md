# 🧠 NLP Assignment – Sequence Generation Tasks

This repository contains implementations for two major sequence generation tasks:

1. **Seq2Seq Encoder-Decoder Model using the Dakshina Dataset**  
2. **Fine-tuning GPT-2 for Poem/Song Lyric Generation**

---

## 📁 Task 1: Seq2Seq Model – Dakshina Dataset

### 📥 Dataset

The **Dakshina Dataset** is a multilingual dataset used for transliteration tasks.  
To access the dataset:

1. Download from the [official source](https://www.microsoft.com/en-us/download/details.aspx?id=100613) or use provided files.
2. Upload to your **Google Drive**.
3. Mount Google Drive in your Colab environment using:

```python
from google.colab import drive
drive.mount('/content/drive')

file_path = '/content/drive/MyDrive/dakshina_dataset_v1.0.tar'
```

---

### 🧠 Model Architecture

The model is a **Sequence-to-Sequence (Seq2Seq)** architecture using **LSTM** (by default):

#### 🔹 Encoder
- Embeds input token sequences into dense vectors.
- Uses an LSTM RNN to process these embeddings.
- Outputs the final hidden and cell states.

#### 🔹 Decoder
- Accepts target token sequences (shifted by one step).
- Embeds tokens similarly to the encoder.
- Uses another LSTM to process embeddings and encoder's state.
- Outputs token probabilities via a Dense + Softmax layer.

> ⚙️ You can customize the RNN cell by setting `cell_type='GRU'` or `cell_type='SimpleRNN'`.

#### 🔢 Parameters
- Over **200,000 trainable parameters**
- Supports dynamic embedding dimensions and hidden units

---

## 📝 Task 2: Fine-Tuning GPT-2 – Poetry & Lyrics Generation

### 📥 Dataset

Dataset used: [Poetry Dataset from Kaggle](https://www.kaggle.com/paultimothymooney/poetry)

- All `.txt` poem files are read into a single list.
- Used to create a **Hugging Face Dataset** for easy training and evaluation.

---

### 🤖 Model & Tokenizer

- **Model**: `GPT-2` from Hugging Face's model hub.
- **Tokenizer**: `GPT2Tokenizer`
  - `pad_token` is set to `eos_token` to avoid padding issues.
  
- Task: Fine-tune GPT-2 using **masked language modeling** on the tokenized dataset.

---

### ⚙️ Running the Script

1. Install necessary libraries: `transformers`, `datasets`, `torch`, etc.
2. Run on **Google Colab with GPU** enabled for faster training.
3. Monitor loss and generate samples at intervals.

---

## ✅ Conclusion

This project demonstrates:
- Training a **Seq2Seq Encoder-Decoder** model using the **Dakshina dataset** for transliteration.
- Fine-tuning a **GPT-2 model** to generate poems/lyrics from scratch based on a given prompt.

Both models show how powerful deep learning architectures can be applied to different natural language generation tasks.

---

### 🚀 Future Enhancements
- Integrate beam search or attention mechanisms into the decoder for better sequence generation.
- Experiment with larger transformer models like GPT-Neo.
- Use a validation set to implement early stopping and prevent overfitting.
