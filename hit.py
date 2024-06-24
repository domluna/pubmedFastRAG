import requests


# Function to embed text using embed.py
def get_embedding(text):
    url = "http://localhost:8002/embed"
    payload = {"text": text}
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        return response.json()["binary_embedding"][0]
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


# Main script
if __name__ == "__main__":
    # Example text to embed
    # text = "This is a sample text to embed and find matches for."
    # text = "What is the role of GLP-1 and GLP-1 agonists in losing excess weight?"
    text = "What are the biologies of TEAD?"

    try:
        # Get the binary embedding
        binary_embedding = get_embedding(text)
        print(f"Binary embedding obtained. Length: {len(binary_embedding)}")

        # Find matches using the binary embedding
        matches = find_matches(binary_embedding)
        print("\nMatches found:")
        for match in matches:
            print(f"ID: {match['id']}, Distance: {match['distance']}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
