# Literature Buddy

PDF reader with AI-powered chat and RAG (Retrieval-Augmented Generation).

## Features

- 📖 PDF viewing with drag-and-drop support
- 💬 AI chat about document content
- 🔍 RAG-based retrieval
- 🖍️ Text highlighting
- 🤖 Local model inference (no API keys needed)

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/abubakr-shafique/Literature_Buddy.git
cd Literature_Buddy
```

### 2. Create virtual environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download models

```bash
python download_models.py
```

This downloads ~2.6GB of models to the `./models` directory.

## Usage

### Run the application

```bash
python -m literature_buddy.main
```

Or:

```bash
python src/literature_buddy/main.py
```

### First time setup

1. Run `python download_models.py` to download all models
2. Launch the application
3. Drag and drop a PDF file
4. Start chatting!

## Configuration

Edit `config/settings.yaml` to customize:

- Model paths
- Chunk sizes
- UI settings
- Device (CPU/CUDA)

## Project Structure

```
Literature_Buddy/
├── config/
│ └── settings.yaml # Configuration
├── models/ # Downloaded models (created automatically)
├── src/
│ └── literature_buddy/
│ ├── main.py # Entry point
│ ├── config/ # Settings loader
│ ├── models/ # Model loading & backends
│ ├── document/ # PDF loading & parsing
│ ├── rag/ # RAG pipeline
│ ├── retrieval/ # Vector search
│ └── gui/ # PyQt6 interface
├── requirements.txt
├── download_models.py
└── README.md
```


## Requirements

- Python 3.9+
- 3GB+ disk space for models
- 4GB+ RAM recommended

## License

MIT
