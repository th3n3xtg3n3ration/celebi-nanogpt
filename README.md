# Celebi NanoGPT

A lightweight, practical implementation of a GPT-style language model for experimentation, learning, and rapid prototyping.

## Overview

**Celebi NanoGPT** is designed to provide a minimal and approachable foundation for training and evaluating transformer-based language models with PyTorch. The project focuses on simplicity, readability, and extensibility so developers can quickly adapt it for custom datasets and workflows.

## Features

- Minimal GPT-style architecture for educational and experimental use
- PyTorch-based training workflow
- YAML-friendly configuration patterns
- Progress tracking utilities for long-running jobs
- Easy dependency setup with a small requirements footprint

## Tech Stack

- Python
- PyTorch
- Hugging Face Datasets
- TQDM
- PyYAML

## Installation

1. Clone the repository:
   ```bash
   git clone <your-repository-url>
   cd celebi-nanogpt
   ```

2. (Optional but recommended) Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirement.txt
   ```

## Quick Start

> The current repository provides core dependencies and project scaffolding. If you are extending the training pipeline, add your model, dataset, and training scripts under a clear `src/` structure (for example: `src/model.py`, `src/train.py`, `configs/train.yaml`).

Suggested run pattern once scripts are added:

```bash
python src/train.py --config configs/train.yaml
```

## Project Structure

Current top-level files:

```text
celebi-nanogpt/
├── LICENSE
├── README.md
└── requirement.txt
```

## Roadmap Suggestions

- Add baseline GPT model implementation
- Add tokenizer and text preprocessing pipeline
- Add evaluation metrics (perplexity, validation loss)
- Add reproducible experiment configs
- Add unit tests and CI checks

## Contributing

Contributions are welcome. Please open an issue to discuss substantial changes before submitting a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
