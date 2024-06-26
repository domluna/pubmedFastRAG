from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
import time
import uvicorn
import fire

MATRYOSHKA_DIM = 512

app = FastAPI()

tokenizer = None
model = None


class TextInput(BaseModel):
    text: str
    only_binary: bool = True


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


def embed_text(text: str) -> (np.ndarray, dict):
    global tokenizer, model
    timings = {}

    start_time = time.time()

    with torch.no_grad():
        tokenize_start = time.time()
        inputs = tokenizer(
            text,
            return_tensors="pt",
        ).to(model.device)
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
    embeddings = F.normalize(embeddings, p=2, dim=1).cpu().numpy().reshape(-1)

    process_end = time.time()
    timings["post_processing"] = process_end - process_start

    quantize_start = time.time()
    quantized_embeddings = np.packbits(embeddings > 0)
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
        d = {
            "binary_embedding": binary_embeddings.tolist(),
        }
        if not input.only_binary:
            d["embedding"] = embedding.tolist()
        return d
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def main(port: int = 8002, device: str = "cpu"):
    global tokenizer, model

    # Set the device
    device = torch.device(
        device if torch.cuda.is_available() and device == "cuda" else "cpu"
    )
    print(f"Using device: {device}")

    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(
        "nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True
    )
    model = AutoModel.from_pretrained(
        "nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True
    ).to(device)
    model.eval()

    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    fire.Fire(main)
