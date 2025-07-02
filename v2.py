import requests
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer, util
import numpy as np
import logging
import pandas as pd

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Ensure that the NLTK punkt tokenizer is downloaded
def ensure_nltk_punkt():
    """Ensure that the NLTK 'punkt' tokenizer is downloaded."""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

# Fetch and parse the webpage
def fetch_and_parse_page(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Will raise an HTTPError if the status is 4xx or 5xx
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')  # Assuming the main content is inside <p> tags
        text = " ".join([p.get_text() for p in paragraphs])
        sentences = sent_tokenize(text)
        return sentences
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching {url}: {e}")
        return []

# Compute similarity matrix between queries and the content
def compute_similarity_matrix(queries, content):
    model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
    query_embeddings = model.encode(queries, convert_to_tensor=True)
    content_embeddings = model.encode(content, convert_to_tensor=True)

    sim_matrix = util.pytorch_cos_sim(query_embeddings, content_embeddings)
    return sim_matrix

# Function to analyze URL with detailed scores
def analyze_url_with_detailed_scores(url, queries):
    # Fetch the page content (sentences from paragraphs)
    paragraph_passages = fetch_and_parse_page(url)
    if not paragraph_passages:
        return None
    
    # Compute the similarity matrix between the queries and the content
    sim_matrix = compute_similarity_matrix(queries, paragraph_passages)
    
    # Initialize the Sentence Transformer model for embedding content
    model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
    
    results = []

    for i, query in enumerate(queries):
        query_scores = sim_matrix[i].cpu().numpy()  # Extract the row for the current query
        max_score_idx = np.argmax(query_scores)  # Find the index of the best matching sentence
        
        best_match_score = query_scores[max_score_idx]

        # Compute the similarity score between the current query and the entire content of the page
        overall_content_embedding = model.encode([" ".join(paragraph_passages)], convert_to_tensor=True)
        query_embedding = model.encode([query], convert_to_tensor=True)
        query_vs_content_score = util.pytorch_cos_sim(query_embedding, overall_content_embedding).item()

        # Determine if the content meets the intent
        intent_match = "Yes" if best_match_score >= 0.7 else "No"

        # Collecting query score, best match, and the corresponding content sentence
        results.append({
            "URL": url,
            "Query": query,
            "Best Matching Sentence": paragraph_passages[max_score_idx],
            "Match Score": best_match_score,
            "Query vs Overall Content Score": query_vs_content_score,
            "Intent Match": intent_match
        })
    
    return results

# Function to calculate and rank the most relevant URL
def rank_urls_by_relevance(all_results):
    # Convert the results to a DataFrame
    df = pd.DataFrame(all_results)

    # Calculate the average match score and the percentage of intent matches for each URL
    url_summary = df.groupby('URL').agg(
        avg_match_score=pd.NamedAgg(column='Match Score', aggfunc='mean'),
        intent_match_percentage=pd.NamedAgg(column='Intent Match', aggfunc=lambda x: (x == 'Yes').mean() * 100)
    ).reset_index()

    # Sort URLs by highest avg_match_score and intent_match_percentage
    url_summary_sorted = url_summary.sort_values(by=['avg_match_score', 'intent_match_percentage'], ascending=False)

    # Identify the best intent-matched URL
    best_url = url_summary_sorted.iloc[0]
    
    return url_summary_sorted, best_url

# Function to export results to an Excel file
def export_to_excel(results, ranked_urls, best_url, filename="content_analysis_results.xlsx"):
    try:
        # Convert results to a DataFrame
        df = pd.DataFrame(results)
        
        # Convert ranked URLs to a DataFrame
        ranked_urls_df = ranked_urls
        
        # Create a new Excel writer object and add both DataFrames
        with pd.ExcelWriter(filename) as writer:
            # Write detailed query results
            df.to_excel(writer, sheet_name="Detailed Results", index=False)
            
            # Write ranked URLs
            ranked_urls_df.to_excel(writer, sheet_name="Ranked URLs", index=False)
            
            # Add a summary sheet for the best URL
            summary_df = pd.DataFrame({
                "Best Intent Matched URL": [best_url['URL']],
                "Avg Match Score": [best_url['avg_match_score']],
                "Intent Match Percentage": [best_url['intent_match_percentage']]
            })
            summary_df.to_excel(writer, sheet_name="Best Intent Matched URL", index=False)

        print(f"Results have been successfully saved to {filename}")
    except Exception as e:
        logging.error(f"Error saving to Excel: {e}")

# Main function to interact with the user and process the URLs and queries
def main():
    # Ask the user for input: URLs and queries
    urls_input = input("Enter URLs (comma or line-separated): ").strip()
    queries_input = input("Enter queries (comma or line-separated): ").strip()

    # Split the input into lists
    urls = [url.strip() for url in urls_input.split(",")]
    queries = [query.strip() for query in queries_input.split(",")]

    # Loop over each URL and analyze it with the provided queries
    all_results = []  # This will hold results from all URLs for exporting to Excel
    for url in urls:
        print(f"\nAnalyzing URL: {url}\n")
        results = analyze_url_with_detailed_scores(url, queries)
        
        if results:
            # Append results to all_results for Excel export
            all_results.extend(results)
            
            print("Detailed Results:\n")
            for result in results:
                print(f"URL: {result['URL']}")
                print(f"Query: {result['Query']}")
                print(f"Best Matching Sentence: {result['Best Matching Sentence']}")
                print(f"Match Score: {result['Match Score']}")
                print(f"Query vs Overall Content Score: {result['Query vs Overall Content Score']}")
                print(f"Intent Match: {result['Intent Match']}\n")
        else:
            print(f"Failed to fetch or analyze content from {url}.\n")
    
    # Rank URLs by relevance and get the best URL
    if all_results:
        ranked_urls, best_url = rank_urls_by_relevance(all_results)
        
        # Export all results to an Excel file
        export_to_excel(all_results, ranked_urls, best_url)

if __name__ == "__main__":
    main()

