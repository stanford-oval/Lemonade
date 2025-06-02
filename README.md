<div align="center">
   <h1>
      <img src="assets/lemonade.png" alt="LEMONADE" height="40" style="vertical-align: top; margin-right: 0px;">
      LEMONADE: A Large Multilingual Expert-Annotated Abstractive Event Dataset for the Real World
   </h1>
</div>

<div align="center">
  <img src="assets/stanford.png" alt="Stanford" height="80" style="margin: 10px;">
  <img src="assets/northwestern.png" alt="Northwestern" height="80" style="margin: 10px;">
  <img src="assets/acled.webp" alt="ACLED" height="80" style="margin: 10px;">
</div>

<div align="center">
  <img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/stanford-oval/Lemonade">
</div>

# Setup

This repository contains the code to perform multilingual abstractive event extraction using the LEMONADE dataset. LEMONADE is a large, expert-annotated dataset for event extraction from news articles in 20 languages: English, Spanish, Arabic, French, Italian, Russian, German, Turkish, Burmese, Indonesian, Ukrainian, Korean, Portuguese, Dutch, Somali, Nepali, Chinese, Persian, Hebrew, and Japanese.

The dataset itself is accessible on [🤗 Hugging Face Hub](https://huggingface.co/datasets/stanford-oval/Lemonade).


## Prerequisites

Before setting up the project, you'll need the following:

- **[Pixi](https://pixi.sh/) package manager** - A fast package manager for Python environments. If you don't have Pixi installed, follow the [installation guide](https://pixi.sh/latest/#installation). Pixi will handle all Python dependencies.

This project has been tested with Python 3.12. Pixi will automatically use this version when installing dependencies, so you don't need to worry about Python version management.

- **LLM API key** - You'll need access to a large language model API (OpenAI, Anthropic, etc.) as the pipeline relies on LLM calls for event processing tasks. Alternatively, you can use a local LLM if you have one set up.


## Installation

Setting up the project is straightforward with Pixi. Run these commands in the project directory:

```bash
# Install all dependencies and create the environment
pixi install

# Configure your LLM API key. This variable should have the same name as the one you specify in llm_config.yaml.
export OPENAI_API_KEY=your_api_key_here
```

You also need to configure how to access LLMs. This is done through a configuration file:

Edit the `llm_config.yaml` file to specify which language models to use. Follow the [ChainLite documentation](https://github.com/stanford-oval/chainlite/blob/main/llm_config.yaml) for the configuration format.


# Usage

The pipeline can be run in two modes: event detection (ED) only, or the full pipeline including event detection (ED), abstractive event argument extraction (AEAE), and abstractive entity linking (AEL).

The pipeline uses the [LEMONADE dataset](https://huggingface.co/datasets/stanford-oval/Lemonade) containing multilingual news articles with event annotations. You don't need to download this dataset manually—the script will automatically download it the first time you run the pipeline.

## Run Pipeline

Here's a basic example that processes 5 English articles using GPT-4:

```bash
python -m zest.run_pipeline \
  --output_file ./outputs.jsonl \
  --language en \
  --examples_per_language 5 \
  --task full \
  --engine gpt-4.1-mini \
  --data_split dev
```

## Command-Line Arguments

The pipeline accepts several arguments to customize its behavior:

- `--output_file`: Specifies where to save the processed results in JSONL format. Each line will contain one processed news article with extracted events and entities.

- `--language`: Determines which language to process. The system supports 20 languages: en, es, ar, fr, it, ru, de, tr, my, id, uk, ko, pt, nl, so, ne, zh, fa, he, ja. If not specified, all languages will be processed.

- `--examples_per_language`: Controls how many articles to process per language. Use -1 to process all available articles in the selected languages, or specify a positive number to limit processing.

- `--task`: Selects the task. Use 'event_detection' for event detection only, or 'full' for the complete pipeline including abstractive event argument extraction and abstractive entity linking.

- `--engine`: Specifies which LLM to use. The model must be configured in your `llm_config.yaml` file.

- `--data_split`: Chooses which split of the dataset to process. Use 'dev' for the development set or 'test' for the test set.

- `--entity_database_path`: Points to the entity database file used for entity linking. This JSONL file contains known entities and is required when running the full pipeline. Defaults to 'event_dataset/entity_database.jsonl'.

## Performance Considerations

Processing time and costs depend on the language model used and the amount of data processed. For example, using the `gpt-4.1-mini` model:

- **Runtime**: Processing 500 English events takes around 30 minutes.

- **API Costs**: The same 500-event run costs approximately $31 in LLM API calls. Costs vary significantly between different models and providers.

- **Evaluation**: After processing, the evaluation step takes an additional 20 minutes due to rate limits when geocoding locations through nominatim.org.

- **Caching**: The system caches both LLM outputs (using [Redis](https://redis.io/docs/latest/)) and geocoding results (using [Diskcache](https://grantjenks.com/docs/diskcache/)). This means re-running the same configuration will be much faster and free of charge.



# Citation

If you have used code or data from this repository, please cite the following paper:

```bibtex
@inproceedings{semnani2025lemonade,
   title={{LEMONADE}: A Large Multilingual Expert-Annotated Abstractive Event Dataset for the Real World},
   author={Semnani, Sina J. and Zhang, Pingyue and Zhai, Wanyue and Li, Haozhuo and Beauchamp, Ryan and Billing, Trey and Kishi, Katayoun and Li, Manling and Lam, Monica S.},
   booktitle={Findings of the Association for Computational Linguistics: ACL 2025},
   year={2025}
}
```
