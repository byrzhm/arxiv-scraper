# ArXiv Scraper with LLM Summaries

This tool scrapes ArXiv for papers on specific topics, uses OpenAI to generate bilingual summaries (English and Chinese), and saves the results as markdown files.

## Features

- Search ArXiv for papers matching specific keywords and categories
- Automatically filter papers by relevance to AI infrastructure topics
- Generate concise summaries in English and Chinese using OpenAI's GPT-4
- Save summaries as well-formatted markdown files for easy publishing

## Setup

### Prerequisites

- Python 3.8 or higher
- `uv` package manager (or pip)

### Installation

1. Clone this repository
2. Install core dependencies:

```bash
uv pip install -r requirements.txt
```

3. Create a `.env` file with the API key(s) for your preferred LLM provider(s):

```dotenv
# OpenAI API
OPENAI_API_KEY=your-openai-api-key-here
OPENAI_MODEL=gpt-4  # Optional, default is gpt-4

# DeepSeek API
# DEEPSEEK_API_KEY=your-deepseek-api-key-here
# DEEPSEEK_MODEL=deepseek-chat  # Optional

# Alibaba Qwen API
# QWEN_API_KEY=your-qwen-api-key-here
# QWEN_MODEL=qwen-max  # Optional
```

## Usage

### Command-line Arguments

The script supports several command line arguments:

```bash
python main.py [options]
```

Options:
- `--fetch-only`: Only fetch papers, don't generate summaries
- `--max-results NUMBER`: Maximum number of papers to fetch (default is 50)
- `--keywords "KEYWORDS"`: Search keywords separated by OR
- `--categories "CAT1,CAT2"`: ArXiv categories to search
- `--min-score FLOAT`: Minimum relevance score for filtering (0-1)
- `--provider NAME`: LLM provider to use (default is "openai")
  - Supported providers: "openai", "deepseek", "qwen"
  - See [LLM_PROVIDERS.md](LLM_PROVIDERS.md) for detailed setup instructions

### Examples

Fetch papers and generate summaries using OpenAI:
```bash
python main.py
```

Just fetch papers without generating summaries:
```bash
python main.py --fetch-only
```

Use DeepSeek to summarize papers with specific keywords:
```bash
python main.py --provider deepseek --keywords "transformer OR LLM OR fine-tuning" --categories "cs.LG,cs.CL"
```

Use Qwen with 20 papers maximum:
```bash
python main.py --provider qwen --max-results 20
```

### Default Configuration

The default configuration searches for AI infrastructure related papers:
- Keywords: infrastructure, serving, inference, distributed, training, etc.
- Categories: Machine Learning, AI, Distributed Computing, Neural Networks, etc.

## Customization

- Modify the TARGET_TOPICS list to change which topics are considered relevant
- Adjust the prompts in `summarize_paper` and `summarize_paper_cn` functions
- Modify the post format in the `generate_post` function

## Note

This tool uses the ArXiv API, which has rate limits. The script includes a 1-second delay between requests to be respectful of these limits.