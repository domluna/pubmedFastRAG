import requests


def find_matches(text, k=5):
    # rag.jl endpoint
    url = "http://localhost:8003/find_matches"
    payload = {"query": text, "k": k}
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Error from RAG service: {response.text}")


if __name__ == "__main__":
    text = "What is the role of GLP-1 and GLP-1 agonists in losing excess weight?"
    # text = "What are the biologies of TEAD?"

    matches = find_matches(text)

    for match in matches:
        print(f"ID: {match['pmid']}, Distance: {match['distance']}")
        print(f"Title: {match['title']}")
        print(f"Authors: {match['authors']}")
        print(f"Publication Year: {match['publication_year']}")
        print(f"Abstract: {match['abstract'][:200]}...")
        print()
