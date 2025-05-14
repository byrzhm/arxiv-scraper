# Using Different LLM Providers with ArXiv Scraper

This guide explains how to use different Large Language Model (LLM) providers with the ArXiv Scraper tool.

## Supported LLM Providers

The tool currently supports the following LLM providers:

1. **OpenAI** (default) - GPT-4, GPT-3.5-Turbo
2. **DeepSeek** - DeepSeek Chat
3. **Alibaba Qwen** - Qwen-Max, Qwen-Plus, etc.

## Setup for Different Providers

### Prerequisites

First, install the `openai`:

```bash
uv pip install openai
```

### Provider-specific Configuration

#### OpenAI

1. Get your API key from [OpenAI Platform](https://platform.openai.com/api-keys)
2. Add to your `.env` file:
   ```
   OPENAI_API_KEY=your-api-key-here
   OPENAI_MODEL=gpt-4  # Optional, defaults to gpt-4
   ```
3. Run with: `python main.py --provider openai`

#### DeepSeek

1. Get your API key from [DeepSeek Platform](https://platform.deepseek.com/)
2. Add to your `.env` file:
   ```
   DEEPSEEK_API_KEY=your-api-key-here
   DEEPSEEK_MODEL=deepseek-chat  # Optional
   ```
3. Run with: `python main.py --provider deepseek`

#### Alibaba Qwen

1. Get your API key from [Alibaba Cloud](https://bailian.console.aliyun.com/)
2. Add to your `.env` file:
   ```
   QWEN_API_KEY=your-api-key-here
   QWEN_MODEL=qwen-max  # Optional
   ```
3. Run with: `python main.py --provider qwen`

## Testing Your Setup

You can test if your LLM provider is properly set up by running the test script:

```bash
# Test default provider (OpenAI)
python test_llm_clients.py

# Test a specific provider
python test_llm_clients.py deepseek
```

## Performance Considerations

Different LLM providers have different strengths:

- **OpenAI's GPT-4** - Generally high-quality summaries with good understanding of technical content
- **DeepSeek** - Good performance on technical and scientific content
- **Qwen** - Good performance on bilingual content (English/Chinese)

## Troubleshooting

Common issues and their solutions:

1. **Missing Dependencies**
   ```
   ImportError: No module named 'openai'
   ```
   Solution: Install the required package with `uv pip install openai`

2. **API Key Errors**
   ```
   ValueError: OPENAI_API_KEY not found in environment or .env file
   ```
   Solution: Check that you've added the correct API key to your `.env` file

3. **Rate Limiting**
   If you encounter rate limiting errors, consider:
   - Reducing the number of papers processed
   - Adding a delay between API calls
   - Upgrading to a higher tier API plan

## Extending to Other Providers

To add support for additional LLM providers, you can extend the `llm_clients.py` module by:

1. Adding a new client class that inherits from the `LLMClient` abstract class
2. Implementing the `generate_completion` method for the new provider
3. Updating the `get_llm_client` factory function to support the new provider
