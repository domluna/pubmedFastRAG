from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
import time
import uvicorn

MATRYOSHKA_DIM = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DEVICE", DEVICE)

app = FastAPI()

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(
    "nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True
)
model = AutoModel.from_pretrained(
    "nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True
).to(DEVICE)
model.eval()


class TextInput(BaseModel):
    text: str


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


# Function to embed text snippets
def embed_text(text: str) -> (np.ndarray, dict):
    timings = {}

    start_time = time.time()

    with torch.no_grad():
        tokenize_start = time.time()
        inputs = tokenizer(
            "search_document: " + text,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(DEVICE)
        tokenize_end = time.time()
        timings["tokenization"] = tokenize_end - tokenize_start

        model_start = time.time()
        outputs = model(**inputs)
        model_end = time.time()
        timings["model_inference"] = model_end - model_start

    process_start = time.time()

    embeddings = mean_pooling(outputs, inputs["attention_mask"])
    embeddings = F.layer_norm(embeddings, normalized_shape=(embeddings.shape[1],))
    embeddings = embeddings[:, :MATRYOSHKA_DIM]
    embeddings = F.normalize(embeddings, p=2, dim=1).cpu().numpy()

    process_end = time.time()
    timings["post_processing"] = process_end - process_start

    quantize_start = time.time()
    quantized_embeddings = np.packbits(embeddings > 0).reshape(embeddings.shape[0], -1)
    quantize_end = time.time()
    timings["quantization"] = quantize_end - quantize_start

    total_time = time.time() - start_time
    timings["total"] = total_time

    return embeddings, quantized_embeddings, timings


@app.post("/embed")
async def embed(input: TextInput):
    try:
        embedding, binary_embeddings, timings = embed_text(input.text)
        print(timings)

        return {
            "embedding": embedding.tolist(),
            "binary_embedding": binary_embeddings.tolist(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)
