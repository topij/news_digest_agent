"""
news_digest_agent.py – AI-powered news digest generator.

This script creates a daily digest of AI-related news articles by:
1. Fetching recent articles from NewsAPI
2. Analyzing them for key themes and trends
3. Generating a structured summary with implications and future outlook

The script uses PydanticAI agents for:
- News fetching and validation
- Thematic analysis
- Article summarization (experimental)

Dependencies:
- NewsAPI for article fetching
- OpenAI API for analysis and summarization
- Rich for terminal output formatting

Environment Variables:
- NEWS_API_KEY: Your NewsAPI key
- OPENAI_API_KEY: Your OpenAI API key

Usage:
    python news_digest_agent.py
"""

import os
import asyncio
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Literal, AsyncGenerator
from contextlib import asynccontextmanager

from pydantic import BaseModel, ConfigDict, TypeAdapter
from pydantic_ai import Agent, RunContext
from newsapi import NewsApiClient
from dotenv import load_dotenv, find_dotenv
from rich.console import Console
from rich.spinner import Spinner
from rich.panel import Panel
from rich.live import Live
from rich.progress import Progress, SpinnerColumn, TextColumn

import requests
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.ERROR)
console = Console()

#---------------------- uncomment if using logfire ---------------------------
# Logfire is a tool for logging and monitoring AI agents.
# import logfire
# logfire.configure(
#     service_name="daily-digest",
#     token="place_logfire_token_here",
# )
# logfire.instrument_pydantic_ai()

# ---------------------------------------------------------------------------
# 0.  ENV / LOAD
# ---------------------------------------------------------------------------

# Debug if problems with environment variables. 
# Set logger to INFO for more verbose output.
logger.info("Starting daily digest agent")
logger.info(f"Current working directory: {os.getcwd()}")
dotenv_path = find_dotenv(usecwd=True)
logger.info(f"Looking for .env file at: {dotenv_path}")
load_dotenv(dotenv_path)
logger.info("Environment variables loaded: %s", {k: v for k, v in os.environ.items() if k in ['OPENAI_API_KEY', 'NEWS_API_KEY']})

if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError(
        "OPENAI_API_KEY environment variable is missing. "
        "Set it (or pass api_key to the Agent) before running."
    )

# ---------------------------------------------------------------------------
# 1.  PYDANTIC SCHEMAS
# ---------------------------------------------------------------------------

class NewsArticle(BaseModel):
    """
    Schema for a single news article.
    
    Attributes:
        title: The article's headline
        description: A brief description or summary of the article
        url: The full URL to the article
        source: The name of the news source
        published_at: When the article was published
        author: The article's author (if available)
    """
    title: str
    description: str | None = None
    url: str
    source: str
    published_at: datetime
    author: str | None = None


class NewsResponse(BaseModel):
    """
    Schema for the NewsAPI response.
    
    Attributes:
        articles: List of news articles
        total_results: Total number of articles found
        status: API response status ('ok' or error)
    """
    articles: List[NewsArticle]
    total_results: int
    status: str


class NewsDeps(BaseModel):
    """
    Dependencies for the news digest agent.
    
    Attributes:
        api_key: NewsAPI key
        client: Configured NewsAPI client instance
    """
    api_key: str
    client: NewsApiClient
    model_config = ConfigDict(arbitrary_types_allowed=True)


class ThematicSummary(BaseModel):
    """
    Schema for thematic analysis output.
    
    Attributes:
        themes: List of identified key themes
        analysis: Detailed analysis of the themes
        implications: Current implications of the themes
        future_outlook: Predictions about future developments
    """
    themes: List[str]
    analysis: str
    implications: str
    future_outlook: str


class SummarisedArticle(BaseModel):
    """
    Schema for article summarization output.
    
    Attributes:
        title: The article's headline
        url: The article's URL
        summary: A concise summary (≤150 words)
        model: The model used for summarization
    """
    title: str
    url: str
    summary: str  # ≤150 words, plain-English
    model: Literal["openai:gpt-4.1-mini"] = "openai:gpt-4.1-mini"


# ---------------------------------------------------------------------------
# 2.  PROMPT TEMPLATES
# ---------------------------------------------------------------------------

PROMPTS = {
    "digest": (
        "You are a news-digest assistant. "
        "Use the registered tools to fetch news and create thematic summaries. "
        "Always produce schema-valid JSON that matches NewsResponse."
    ),
    "thematic": (
        "You are an AI news analyst specializing in thematic analysis. "
        "Given a set of news articles, identify key themes, analyze their significance, "
        "and provide insights about future implications. "
        "Structure your response according to the ThematicSummary schema."
    ),
    "thematic_analysis": (
        "Analyze these news articles and provide a thematic summary that:\n"
        "1. Identifies the 3-4 most prominent themes\n"
        "2. Analyzes their significance and implications\n"
        "3. Provides insights about future developments\n\n"
        "Articles to analyze:\n{articles_text}"
    ),
    "summariser": (
        "You are a concise news summariser. "
        "Given an article's TITLE, URL and raw HTML, produce a plain‑English "
        "abstract no longer than 150 words. "
        "Do not add opinions or extra metadata; return only fields required by "
        "SummarisedArticle."
    )
}


# ---------------------------------------------------------------------------
# 3.  AGENTS
# ---------------------------------------------------------------------------

def create_agent(name: str, model: str, output_type: type, system_prompt: str) -> Agent:
    """
    Factory function to create agents with consistent configuration.
    
    Args:
        name: Identifier for the agent
        model: The OpenAI model to use
        output_type: Pydantic model for output validation
        system_prompt: The system prompt for the agent
    
    Returns:
        A configured PydanticAI Agent instance
    """
    return Agent(
        model,
        output_type=output_type,
        system_prompt=system_prompt,
    )


# Create agents using the factory function
digest_agent = create_agent(
    "digest",
    "openai:gpt-4.1-mini",
    NewsResponse,
    PROMPTS["digest"]
)

thematic_agent = create_agent(
    "thematic",
    "openai:gpt-4.1-mini",
    ThematicSummary,
    PROMPTS["thematic"]
)

summariser_agent = create_agent(
    "summariser",
    "openai:gpt-4.1-mini",
    SummarisedArticle,
    PROMPTS["summariser"]
)


@digest_agent.tool
async def search_news(
    ctx: RunContext[NewsDeps],
    query: str,
    from_date: str | None = None,
    to_date: str | None = None,
    language: str = "en",
    sort_by: str = "publishedAt",
    page_size: int = 10,
) -> NewsResponse:
    """
    Fetch headlines from NewsAPI and return them as a validated NewsResponse.
    
    Args:
        ctx: The run context containing dependencies
        query: Search query string
        from_date: Start date for news search (YYYY-MM-DD)
        to_date: End date for news search (YYYY-MM-DD)
        language: Language code for articles
        sort_by: Sort order ('relevancy', 'popularity', 'publishedAt')
        page_size: Number of articles to return (max 100)
    
    Returns:
        A validated NewsResponse containing the articles
    
    Raises:
        RuntimeError: If the NewsAPI request fails
    """
    raw: dict = await asyncio.to_thread(
        ctx.deps.client.get_everything,
        q=query,
        from_param=from_date,
        to=to_date,
        language=language,
        sort_by=sort_by,
        page_size=page_size,
    )

    if raw.get("status") != "ok":
        raise RuntimeError(raw.get("message", "Unknown NewsAPI error"))

    ta = TypeAdapter(List[NewsArticle])
    articles = ta.validate_python(
        [
            {
                "title": a["title"],
                "description": a["description"],
                "url": a["url"],
                "source": a["source"]["name"],
                "published_at": a["publishedAt"],
                "author": a["author"],
            }
            for a in raw["articles"]
        ]
    )

    return NewsResponse(
        articles=articles,
        total_results=raw["totalResults"],
        status=raw["status"],
    )


@digest_agent.tool
async def create_thematic_summary(
    ctx: RunContext[NewsDeps],
    articles: List[NewsArticle],
) -> ThematicSummary:
    """
    Create a thematic summary of the articles using the thematic agent.
    
    Args:
        ctx: The run context containing dependencies
        articles: List of articles to analyze
    
    Returns:
        A ThematicSummary containing identified themes and analysis
    """
    articles_text = "\n\n".join([
        f"Article {i+1}:\n"
        f"Title: {art.title}\n"
        f"Description: {art.description or 'No description available'}\n"
        f"Source: {art.source}"
        for i, art in enumerate(articles)
    ])

    result = await thematic_agent.run(
        PROMPTS["thematic_analysis"].format(articles_text=articles_text),
        usage=ctx.usage,
    )
    return result.output


@digest_agent.tool
async def summarise_url(
    ctx: RunContext[NewsDeps],
    title: str,
    url: str,
) -> Optional[SummarisedArticle]:
    """
    Fetch the article body and delegate to the summariser agent for a ≤150‑word abstract.
    
    Note: This is experimental and may fail due to paywalls or access restrictions.
    
    Args:
        ctx: The run context containing dependencies
        title: The article's title
        url: The article's URL
    
    Returns:
        A SummarisedArticle if successful, None if fetching or summarization fails
    """
    def _fetch() -> str:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        return resp.text

    try:
        html = await asyncio.to_thread(_fetch)
    except Exception as e:
        logger.error(f"Failed to fetch article from {url}: {str(e)}")
        return None

    prompt = (
        f"TITLE: {title}\n"
        f"URL: {url}\n\n"
        f"HTML:\n{html}\n\n"
        "Return a concise summary:"
    )

    try:
        result = await summariser_agent.run(prompt, usage=ctx.usage)
        return result.output
    except Exception as e:
        logger.error(f"Failed to summarize article {title}: {str(e)}")
        return None


# ---------------------------------------------------------------------------
# 4.  OUTPUT FORMATTING
# ---------------------------------------------------------------------------

class ProgressManager:
    """
    Helper class to manage progress indicators in the terminal.
    
    This class provides a context manager for showing progress spinners
    and updating their status messages.
    """
    
    def __init__(self, console: Console):
        """
        Initialize the progress manager.
        
        Args:
            console: Rich console instance for output
        """
        self.console = console
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        )
    
    def __enter__(self):
        """Start the progress display."""
        self.progress.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop the progress display."""
        self.progress.stop()
    
    def add_task(self, description: str) -> int:
        """
        Add a new progress task.
        
        Args:
            description: Initial status message
        
        Returns:
            Task ID for updating the status
        """
        return self.progress.add_task(f"[blue]{description}", total=None)
    
    def update_task(self, task_id: int, description: str) -> None:
        """
        Update a progress task's description.
        
        Args:
            task_id: ID of the task to update
            description: New status message
        """
        self.progress.update(task_id, description=f"[blue]{description}")


class OutputFormatter:
    """
    Helper class to format console output consistently.
    
    This class provides methods for printing different types of output
    with consistent styling and formatting.
    """
    
    def __init__(self, console: Console):
        """
        Initialize the output formatter.
        
        Args:
            console: Rich console instance for output
        """
        self.console = console
    
    def print_success(self, message: str) -> None:
        """
        Print a success message with a checkmark.
        
        Args:
            message: The message to display
        """
        self.console.print(f"\n[bold green]✓[/bold green] {message}")
    
    def print_warning(self, message: str) -> None:
        """
        Print a warning message in yellow.
        
        Args:
            message: The warning message
        """
        self.console.print(f"[yellow]{message}[/yellow]")
    
    def print_error(self, message: str) -> None:
        """
        Print an error message in red.
        
        Args:
            message: The error message
        """
        self.console.print(f"[bold red]Error:[/bold red] {message}")
    
    def print_section(self, title: str) -> None:
        """
        Print a section header in blue.
        
        Args:
            title: The section title
        """
        self.console.print(f"\n[bold blue]{title}[/bold blue]")
    
    def print_article(self, article: NewsArticle) -> None:
        """
        Print a single article with its details.
        
        Args:
            article: The article to display
        """
        self.console.print(f"• {article.title}  ({article.source})")
        if article.description:
            self.console.print(f"   ↳ {article.description}")
        self.console.print()
    
    def print_thematic_summary(self, summary: ThematicSummary) -> None:
        """
        Print the thematic summary in a structured format.
        
        Args:
            summary: The thematic summary to display
        """
        self.print_section("Key Themes:")
        for theme in summary.themes:
            self.console.print(f"• {theme}")
        
        self.print_section("Analysis:")
        self.console.print(summary.analysis)
        
        self.print_section("Implications:")
        self.console.print(summary.implications)
        
        self.print_section("Future Outlook:")
        self.console.print(summary.future_outlook)


# Initialize formatters
progress = ProgressManager(console)
output = OutputFormatter(console)


# ---------------------------------------------------------------------------
# 5.  MAIN WORKFLOW
# ---------------------------------------------------------------------------

async def main() -> None:
    """
    Main function to run the news digest workflow.
    
    This function:
    1. Initializes the NewsAPI client
    2. Fetches recent AI policy news
    3. Displays the articles
    4. Generates and displays a thematic summary
    """
    with progress as p:
        init_task = p.add_task("Initializing...")
        deps = NewsDeps(
            api_key=os.environ["NEWS_API_KEY"],
            client=NewsApiClient(api_key=os.environ["NEWS_API_KEY"]),
        )

        yesterday = (datetime.utcnow() - timedelta(days=1)).strftime("%Y-%m-%d")
        p.update_task(init_task, "Fetching news articles...")
        result = await digest_agent.run(
            f"Return the 100 most important AI policy headlines since {yesterday}",
            deps=deps,
        )

    output.print_success(f"Found {result.output.total_results} articles")
    if not result.output.articles:
        output.print_warning("No articles were returned.")
        return

    for article in result.output.articles[:5]:
        output.print_article(article)

    output.print_section("Generating Thematic Summary...")
    with progress as p:
        analysis_task = p.add_task("Analyzing articles...")
        try:
            articles_to_summarize = result.output.articles[:100]
            p.update_task(analysis_task, "Creating thematic summary...")
            summary = await create_thematic_summary(
                ctx=RunContext(deps=deps, model=digest_agent.model, usage={}, prompt=""),
                articles=articles_to_summarize
            )
            output.print_thematic_summary(summary)
        except Exception as e:
            logger.error(f"Error creating thematic summary: {str(e)}")
            output.print_error("Failed to create thematic summary")


def run_digest_agent():
    """
    Entry point that works in both CLI and interactive environments.
    
    This function handles different execution contexts:
    - Regular script execution
    - Jupyter notebooks
    - Interactive Python shells
    - IPython environments
    
    Returns:
        The result of the main workflow
    """
    try:
        # Check if we're in an interactive environment
        import sys
        if hasattr(sys, 'ps1'):  # We're in an interactive shell
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're in a Jupyter notebook or similar
                return asyncio.create_task(main())
            else:
                # We're in a regular interactive shell
                return loop.run_until_complete(main())
        else:
            # We're in a script
            return asyncio.run(main())
    except RuntimeError:
        # Fallback for when no event loop exists
        return asyncio.run(main())


if __name__ == "__main__":
    run_digest_agent()