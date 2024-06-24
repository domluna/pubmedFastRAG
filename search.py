import requests
import sqlite3


# Function to embed text using embed.py
def get_embedding(text):
    url = "http://localhost:8002/embed"
    payload = {"text": text}
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        return response.json()["binary_embedding"]
    else:
        raise Exception(f"Error from embed service: {response.text}")


# Function to find matches using rag.jl
def find_matches(binary_embedding, k=20):
    url = "http://localhost:8003/find_matches"
    payload = {"query": binary_embedding, "k": k}
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Error from RAG service: {response.text}")


def get_article_data(pmids):
    conn = sqlite3.connect("databases/pubmed_data.db")
    cursor = conn.cursor()

    # Create a placeholder string for the SQL query
    placeholders = ",".join(["?" for _ in pmids])

    # Execute a single query to fetch all data at once
    query = f"SELECT pmid, title, authors, abstract, publication_year FROM articles WHERE pmid IN ({placeholders})"
    print(query)
    cursor.execute(
        query,
        pmids,
    )

    # Fetch all results
    results = cursor.fetchall()

    # Create a dictionary to store the results
    article_data = {}

    # Process the results
    for row in results:
        pmid, title, authors, abstract, publication_year = row
        article_data[pmid] = {
            "title": title if title is not None else "Title not found",
            "authors": authors if authors is not None else "Authors not found",
            "abstract": abstract if abstract is not None else "Abstract not found",
            "publication_year": publication_year
            if publication_year is not None
            else "Year not found",
        }

    conn.close()
    return article_data


# Main script
if __name__ == "__main__":
    # Example text to embed
    # text = "This is a sample text to embed and find matches for."
    text = "What is the role of GLP-1 and GLP-1 agonists in losing excess weight?"
    # text = "What are the biologies of TEAD?"

    # Get the binary embedding
    binary_embedding = get_embedding(text)
    print(f"Binary embedding obtained. Length: {len(binary_embedding)}")

    # Find matches using the binary embedding
    matches = find_matches(binary_embedding, 10)
    print("\nMatches found:")

    # Get PMIDs from matches
    pmids = [str(match["id"]) for match in matches]

    # Get abstracts for the matched PMIDs
    article_data = get_article_data(pmids)

    for match in matches:
        pmid = str(match["id"])
        if pmid not in article_data:
            print(f"PMID {pmid} not in database")
            continue
        print(f"ID: {pmid}, Distance: {match['distance']}")
        print(f"Title: {article_data[pmid]['title']}")
        print(f"Authors: {article_data[pmid]['authors']}")
        print(f"Publication Year: {article_data[pmid]['publication_year']}")
        print(f"Abstract: {article_data[pmid]['abstract'][:200]}...")
        print()
