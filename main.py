import feedparser
from datetime import datetime
from pathlib import Path
import urllib.parse
import re
from llm_clients import get_llm_client
from arxiv_categories import ARXIV_CATEGORY_CODES


# Configuration
MAX_RESULTS = 5  # Fetch more to filter later
KEYWORDS = "infrastructure OR serving OR inference OR distributed OR training OR optimization OR efficiency OR systems OR deployment"
CATEGORIES = ",".join(
    [
        ARXIV_CATEGORY_CODES["Machine Learning"],  # cs.LG
        ARXIV_CATEGORY_CODES["Artificial Intelligence"],  # cs.AI
        ARXIV_CATEGORY_CODES["Distributed Parallel and Cluster Computing"],  # cs.DC
        ARXIV_CATEGORY_CODES["Neural and Evolutionary Computing"],  # cs.NE
        ARXIV_CATEGORY_CODES["Computation and Language"],  # cs.CL
        ARXIV_CATEGORY_CODES["Computer Vision and Pattern Recognition"],  # cs.CV
    ]
)
DATE_TAG = datetime.now().strftime("%Y-%m-%d")
MIN_RELEVANCE_SCORE = 0.2  # Lower the threshold since we'll filter manually anyway

# Target topics for relevance filtering
TARGET_TOPICS = [
    "machine learning",
    "deep learning",
    "neural network",
    "AI",
    "artificial intelligence",
    "infrastructure",
    "training",
    "inference",
    "optimization",
    "system",
    "distributed",
    "model",
    "algorithm",
]


# Fetch recent papers
def fetch_recent_papers(query, max_results=10, categories=None):
    # https://info.arxiv.org/help/api/index.html
    base_url = "http://export.arxiv.org/api/query?"

    # Let's try a simpler approach that's more compatible with ArXiv API
    # Build the query with each term separately
    query_terms = query.split(" OR ")

    if not categories:
        categories = ""

    # We'll collect papers from different queries
    all_papers = []

    # Try each category separately for broader coverage
    category_list = categories.split(",")

    for category in category_list:
        category = category.strip()
        if not category:
            continue

        # For each category, try with each query term
        for term in query_terms:
            term = term.strip()
            if not term:
                continue

            # Create a simpler query for this term and category
            search_query = f"search_query=all:{urllib.parse.quote(term)}+AND+cat:{category}&start=0&max_results={max_results}&sortBy=submittedDate&sortOrder=descending"

            try:
                full_url = base_url + search_query
                print(f"Requesting: {full_url}")
                feed = feedparser.parse(full_url)

                # Check if we got entries
                if feed.entries:
                    print(f"Found {len(feed.entries)} papers for {term} in {category}")
                    all_papers.extend(feed.entries)
                else:
                    print(f"No papers found for {term} in {category}")

                # Be nice to the API - avoid rate limits
                import time

                time.sleep(1)

            except Exception as e:
                print(f"Error fetching papers for {term} in {category}: {e}")

    # Remove duplicates based on ID
    unique_papers = []
    seen_ids = set()

    for paper in all_papers:
        if hasattr(paper, "id") and paper.id not in seen_ids:
            seen_ids.add(paper.id)
            unique_papers.append(paper)

    print(f"Total unique papers found: {len(unique_papers)}")
    return unique_papers


# Calculate paper relevance to our target topics
def calculate_relevance_score(paper):
    """Calculate a simple relevance score for a paper based on keyword matching in title and abstract."""
    if not hasattr(paper, "title") or not hasattr(paper, "summary"):
        return 0.0

    text = (paper.title + " " + paper.summary).lower()
    matches = 0

    # Count how many target topics appear in the title or abstract
    for topic in TARGET_TOPICS:
        # Look for the exact phrase
        if topic.lower() in text:
            matches += 1.0
        # Also check for partial matches (individual words)
        else:
            words = topic.lower().split()
            partial_matches = sum(
                1 for word in words if re.search(r"\b" + re.escape(word) + r"\b", text)
            )
            if partial_matches > 0:
                matches += partial_matches / len(words) * 0.5

    # Normalize the score to 0-1 range
    score = min(matches / len(TARGET_TOPICS), 1.0)
    return score


# Filter papers by relevance
def filter_papers_by_relevance(papers, min_score=0.3):
    """Filter and sort papers by relevance score."""
    scored_papers = []

    for paper in papers:
        score = calculate_relevance_score(paper)
        if score >= min_score:
            # Add the score as an attribute
            paper.relevance_score = score
            scored_papers.append(paper)

    # Sort by relevance score, highest first
    return sorted(scored_papers, key=lambda p: p.relevance_score, reverse=True)


# Global LLM client
llm_client = None


# Summarize in English
def summarize_paper(title, abstract):
    prompt = f"""Please summarize the following AI infrastructure paper for a general technical audience:

Title: {title}

Abstract: {abstract}

Summary:"""
    try:
        global llm_client
        return llm_client.generate_completion(prompt)
        # return prompt  # for testing purposes
    except Exception as e:
        print(f"Error summarizing paper in English: {e}")
        return "Error generating summary."


# Summarize in Chinese
def summarize_paper_cn(title, abstract):
    prompt = f"""请用简体中文总结以下 AI 基础设施论文，适合技术社区观众阅读：

标题：{title}

摘要：{abstract}

总结："""
    try:
        global llm_client
        return llm_client.generate_completion(prompt)
        # return prompt  # for testing purposes
    except Exception as e:
        print(f"Error summarizing paper in Chinese: {e}")
        return "生成摘要时出错。"


# Generate bilingual post
def generate_post(papers):
    en_post = f"# \U0001f52c Weekly AI Infrastructure Papers ({DATE_TAG})\n\n"
    zh_post = f"# \U0001f52c 每周 AI 基础设施论文精选（{DATE_TAG}）\n\n"

    for i, paper in enumerate(papers):
        # Clean the title (ArXiv titles often have newlines and extra spaces)
        clean_title = paper.title.replace("\n", " ").strip()

        # Get the paper abstract
        abstract = (
            paper.summary if hasattr(paper, "summary") else "No abstract available"
        )

        # Get the paper link - Use the first arxiv.org link from the links attribute
        paper_link = ""
        if hasattr(paper, "links") and len(paper.links) > 0:
            for link in paper.links:
                if "arxiv.org" in link.href and "pdf" not in link.href:
                    paper_link = link.href
                    break

            # If no appropriate link was found, use the first one
            if not paper_link and len(paper.links) > 0:
                paper_link = paper.links[0].href
        elif hasattr(paper, "link"):
            paper_link = paper.link

        # Get the paper ID if available
        paper_id = paper.id.split("/")[-1] if hasattr(paper, "id") else ""

        print(f"Processing paper {i+1}/{len(papers)}: {clean_title}")

        try:
            en_summary = summarize_paper(clean_title, abstract)
            zh_summary = summarize_paper_cn(clean_title, abstract)

            en_post += f"## {i+1}. {clean_title}\n"
            en_post += f"[{paper_link}]({paper_link})\n\n"
            if paper_id:
                en_post += f"arXiv ID: {paper_id}\n\n"
            en_post += f"**Summary**:\n{en_summary}\n\n"

            zh_post += f"## {i+1}. {clean_title}\n"
            zh_post += f"[{paper_link}]({paper_link})\n\n"
            if paper_id:
                zh_post += f"arXiv ID: {paper_id}\n\n"
            zh_post += f"**总结**：\n{zh_summary}\n\n"
        except Exception as e:
            print(f"Error summarizing paper: {e}")
            en_post += f"## {i+1}. {clean_title}\n"
            en_post += f"[{paper_link}]({paper_link})\n\n"
            en_post += f"**Summary**: Error generating summary.\n\n"

            zh_post += f"## {i+1}. {clean_title}\n"
            zh_post += f"[{paper_link}]({paper_link})\n\n"
            zh_post += f"**总结**：生成摘要时出错。\n\n"

    return en_post, zh_post


# Save posts
def save_posts(en, zh):
    Path("posts").mkdir(exist_ok=True)
    with open(f"posts/weekly_ai_infra_en_{DATE_TAG}.md", "w") as f:
        f.write(en)
    with open(f"posts/weekly_ai_infra_zh_{DATE_TAG}.md", "w") as f:
        f.write(zh)


# Parse command-line arguments
def parse_arguments():
    import argparse

    parser = argparse.ArgumentParser(
        description="ArXiv paper scraper and bilingual summarizer for AI infrastructure papers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                        # Run with default settings
  python main.py --fetch-only           # Only fetch papers, don't generate summaries
  python main.py --max-results 20       # Fetch up to 20 papers
  python main.py --categories "cs.LG"   # Only search in Machine Learning category
  python main.py --provider deepseek    # Use DeepSeek instead of OpenAI
  
Common ArXiv categories:
  cs.AI  - Artificial Intelligence
  cs.LG  - Machine Learning
  cs.CL  - Computation and Language (NLP)
  cs.CV  - Computer Vision
  cs.DC  - Distributed Computing
  cs.NE  - Neural and Evolutionary Computing
  cs.IR  - Information Retrieval
        """,
    )
    parser.add_argument(
        "--fetch-only",
        action="store_true",
        help="Only fetch papers, do not generate summaries",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=MAX_RESULTS,
        help=f"Maximum number of papers to fetch per query (default: {MAX_RESULTS})",
    )
    parser.add_argument(
        "--keywords",
        type=str,
        default=KEYWORDS,
        help='Search keywords, separated by "OR" (default: infrastructure, serving, etc.)',
    )
    parser.add_argument(
        "--categories",
        type=str,
        default=CATEGORIES,
        help=f"ArXiv categories to search, comma-separated (default: {CATEGORIES})",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=MIN_RELEVANCE_SCORE,
        help=f"Minimum relevance score for filtering, 0-1 (default: {MIN_RELEVANCE_SCORE})",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="openai",
        help="LLM provider to use: openai, deepseek, or qwen (default: openai)",
    )

    return parser.parse_args()


# Main flow
def main():
    args = parse_arguments()

    # Override global configuration with command line arguments
    max_results = args.max_results
    keywords = args.keywords
    categories = args.categories
    min_score = args.min_score
    fetch_only = args.fetch_only
    provider = args.provider

    # Only initialize LLM client if we're not in fetch-only mode
    if not fetch_only:
        # Set up the LLM client
        try:
            global llm_client
            llm_client = get_llm_client(provider)
            print(f"Using {provider.capitalize()} as the LLM provider")
        except ImportError as e:
            missing_module = str(e).split("'")[1]
            print(
                f"Warning: Required package '{missing_module}' for {provider} not installed."
            )
            print(f"Please install it with: uv pip install {missing_module}")
            return
        except ValueError as e:
            print(f"Warning: {e}")
            print("Please check your .env file for the required API keys.")
            return
        except Exception as e:
            print(f"Error setting up LLM client: {e}")
            return

    print(f"Fetching papers with keywords: {keywords}")
    print(f"Categories: {categories}")
    papers = fetch_recent_papers(keywords, max_results, categories)

    if not papers:
        print("No papers found. Please check your search terms and try again.")
        return

    print(f"Found {len(papers)} papers from ArXiv. Filtering by relevance...")

    # Filter papers by relevance
    relevant_papers = filter_papers_by_relevance(papers, min_score)
    print(f"After filtering: {len(relevant_papers)} relevant papers")

    # Use a maximum of 10 most relevant papers
    papers_to_process = relevant_papers[:10]

    # Print paper details to check what we fetched
    for i, paper in enumerate(papers_to_process):
        print(f"\nPaper {i+1}:")
        print(f"Title: {paper.title}")
        if hasattr(paper, "relevance_score"):
            print(f"Relevance: {paper.relevance_score:.2f}")
        if hasattr(paper, "id"):
            print(f"ID: {paper.id}")
        if hasattr(paper, "published"):
            print(f"Published: {paper.published}")
        if hasattr(paper, "links") and len(paper.links) > 0:
            print(f"Link: {paper.links[0].href}")
        print("-" * 40)

    if fetch_only:
        print("Fetch-only mode. Skipping summary generation.")
        return

    if not papers_to_process:
        print("No relevant papers found after filtering.")
        return

    print(f"Generating summaries for {len(papers_to_process)} papers...")
    en_post, zh_post = generate_post(papers_to_process)
    save_posts(en_post, zh_post)
    print(
        f"Posts saved in posts/weekly_ai_infra_en_{DATE_TAG}.md and posts/weekly_ai_infra_zh_{DATE_TAG}.md"
    )


if __name__ == "__main__":
    main()
