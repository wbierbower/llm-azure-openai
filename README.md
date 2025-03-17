# llm-azure-openai

[![PyPI](https://img.shields.io/pypi/v/llm-azure-openai.svg)](https://pypi.org/project/llm-azure-openai/)
[![Changelog](https://img.shields.io/github/v/release/wbierbower/llm-azure-openai?include_prereleases&label=changelog)](https://github.com/wbierbower/llm-azure-openai/releases)
[![Tests](https://github.com/wbierbower/llm-azure-openai/workflows/Test/badge.svg)](https://github.com/wbierbower/llm-azure-openai/actions?query=workflow%3ATest)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/wbierbower/llm-azure-openai/blob/main/LICENSE)

API access to Microsoft's Azure OpenAI models

## Installation

Install this plugin in the same environment as [LLM](https://llm.datasette.io/).

```bash
llm install llm-azure-openai
```

## Usage

Export the environment variables:

```bash
export AZURE_TENANT_ID='...'
export AZURE_CLIENT_ID='...'
export AZURE_CLIENT_SECRET='...'
```

Configure the models you want to use from your deployments

On Mac: ~/Library/Application Support/io.datasette.llm/azure-openai-models.yaml

On Linux: ~/.config/io.datasette.llm/azure-openai-models.yaml

```bash
- model_id: o3-mini
  model_name: o3-mini
  azure_endpoint: https://example.openai.azure.com
  api_version: '2024-12-01-preview'
  aliases: ['azure-o3-mini']
  use_azure_ad: true
- model_id: gpt-35-turbo-blue
  model_name: gpt-35-turbo-blue
  azure_endpoint: https://example.openai.azure.com
  api_version: '2024-02-01'
  aliases: ['azure-gpt-35']
  use_azure_ad: true
```

Now run the model using `-m your-model`, for example:

```bash
llm -m azure-o3-mini "..."
```

## Development

To set up this plugin locally, first checkout the code. Then create a new virtual environment:

```bash
cd llm-azure-openai
python3 -m venv venv
source venv/bin/activate
```

Now install the dependencies and test dependencies:

```bash
llm install -e '.[test]'
```

To run the tests:

```bash
pytest
```

## Security

This package is signed using [sigstore](https://sigstore.dev/) to provide supply chain security. When you install this package from PyPI, you can verify its authenticity by checking the digital signatures.

<!-- This project uses [pytest-recording](https://github.com/kiwicom/pytest-recording) to record Gemini API responses for the tests.

If you add a new test that calls the API you can capture the API response like this:
```bash
PYTEST_GEMINI_API_KEY="$(llm keys get gemini)" pytest --record-mode once
```
You will need to have stored a valid Gemini API key using this command first:
```bash
llm keys set gemini
# Paste key here
``` -->
