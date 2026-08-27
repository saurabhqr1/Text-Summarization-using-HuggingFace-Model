
# Text Summarization using Hugging Face (Pegasus + SAMSum)

A fine-tuned **Pegasus** model for abstractive text summarization, trained on the **SAMSum** conversational dataset, served through a **FastAPI** backend with a simple web UI.

This project is a modernized version of the original tutorial-style repo, updated to run on current Python (3.10/3.11) and current library versions (Transformers 4.46, Datasets 3.1, Accelerate 1.1, Evaluate 0.4), with several bug fixes and a lightweight frontend added on top.

---

## Features

- End-to-end ML pipeline: data ingestion → validation → transformation → training → evaluation
- Fine-tunes `google/pegasus-cnn_dailymail` on the SAMSum dialogue-summarization dataset
- FastAPI backend exposing a `/predict` endpoint
- Simple built-in web page (`templates/index.html`) to paste a conversation and get a summary
- Trainable locally (CPU, slow) or on a free GPU via Google Colab (fast)

---

## Project Structure

```
├── app.py                     # FastAPI app (frontend + /predict + /train routes)
├── main.py                    # Runs the full training pipeline (5 stages)
├── config/config.yaml         # Paths, URLs, and directory configuration
├── params.yaml                # Training hyperparameters
├── templates/index.html       # Web UI for the summarizer
├── src/textSummarizer/
│   ├── config/                # Configuration manager
│   ├── conponents/            # Pipeline stage implementations
│   ├── entity/                # Config dataclasses
│   ├── pipeline/              # Stage runners + prediction pipeline
│   └── utils/                 # Shared helper functions
├── requirements.txt
└── Dockerfile
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/saurabhqr1/Text-Summarization-using-HuggingFace-Model.git
cd Text-Summarization-using-HuggingFace-Model
```

### 2. Create a virtual environment (Python 3.10 or 3.11)

```bash
python -m venv venv
# Windows
.\venv\Scripts\Activate.ps1
# macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Training the Model

Run the full pipeline (data ingestion → validation → transformation → training → evaluation):

```bash
python main.py
```

**Note on hardware:** training Pegasus on CPU is very slow (potentially many hours). Training on a free GPU (e.g. Google Colab with a T4) typically completes in under an hour. See the notes at the bottom of this file for the Colab workflow.

Trained artifacts are saved locally to:
```
artifacts/model_trainer/pegasus-samsum-model/
artifacts/model_trainer/tokenizer/
```

These are excluded from version control (see `.gitignore`) since the model checkpoint is several GB.

---

## Running the App

Once a trained model exists in `artifacts/model_trainer/`:

```bash
python app.py
```

Then open:
```
http://localhost:8080
```

This loads a simple web page where you can paste a conversation and click **Summarize** to get a generated summary. The interactive API docs are also available at `http://localhost:8080/docs`.

---

## Training on Google Colab (Recommended if you don't have a GPU)

1. Zip the project (excluding `venv/` and `artifacts/`) and upload it to Colab
2. Enable a free T4 GPU: **Runtime → Change runtime type → T4 GPU**
3. Extract the zip, `cd` into the folder, and run `pip install -r requirements.txt`
4. Run `python main.py`
5. Zip the resulting `artifacts/model_trainer/pegasus-samsum-model` and `tokenizer` folders and download them
6. Extract into your local project's `artifacts/model_trainer/` folder and run `python app.py` locally

---

## Key Fixes vs. the Original Tutorial Repo

- `evaluation_strategy` → renamed to `eval_strategy` (Transformers 4.46+)
- `Trainer(tokenizer=...)` → renamed to `Trainer(processing_class=...)`
- `datasets.load_metric` (removed) → replaced with `evaluate.load('rouge')`
- Fixed `params.yaml`'s `save_steps: 1e6` (PyYAML parses this as a string, not a float, without a decimal point) → changed to a plain integer
- Fixed a bug where `eval_steps` was mistakenly read from `evaluation_strategy` in the configuration manager
- `ModelTrainer` now actually uses values from `params.yaml` instead of hardcoded duplicates
- Fixed the `DataValiadtion` typo → `DataValidation`
- Added missing `sentencepiece` and `protobuf` dependencies required by the Pegasus tokenizer
- Dockerfile updated from the end-of-life `python:3.8-slim-buster` to `python:3.11-slim`

---

## Tech Stack

- **Model:** Pegasus (`google/pegasus-cnn_dailymail`), fine-tuned on SAMSum
- **Libraries:** Transformers, Datasets, Evaluate, Accelerate, PyTorch
- **Backend:** FastAPI + Uvicorn
- **Frontend:** Plain HTML/CSS/JS (Jinja2-rendered)

---

## License

See [LICENSE](LICENSE).
