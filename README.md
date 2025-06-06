# News Digest AI Agent

An AI-powered news digest generator that creates daily summaries of news articles using PydanticAI agents.

Script was done for learning more about AI agents, and how to use PydanticAI to create them.

## Features

- Fetches recent news articles from NewsAPI based on chosen topic
- Analyzes articles for key themes and trends
- Generates structured summaries with implications and future outlook
- Rich terminal output with progress indicators
- Supports both CLI and interactive environments (Jupyter, IPython)

## Prerequisites

- Python 3.x
- NewsAPI key
- OpenAI API key
- Conda (recommended) or pip

## Installation

1. Clone the repository

2. Create and activate the conda environment:
```bash
conda env create -f environment.yml
conda activate pydantic-news-agent
```

3. Create a `.env` file in the project root with your API keys:
```
NEWS_API_KEY=your_news_api_key
OPENAI_API_KEY=your_openai_api_key
```

## Usage

Run the script directly:
```bash
python news_digest_agent.py
```

The script will:
1. Fetch the most recent AI policy (default topic) news articles
2. Display the top 5 articles
3. Generate and display a thematic summary based on the titles and descriptions

## Output

The script provides:
- Article headlines with sources and descriptions
- Key themes identified in the news
- Analysis of current implications
- Future outlook predictions

## Development

The project uses:
- PydanticAI for agent-based processing
- NewsAPI for article fetching
- OpenAI API for analysis and summarization
- Rich for terminal output formatting

## License

MIT License 