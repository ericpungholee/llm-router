"""Pinned, local, response-free encoder and reproducible prompt embedding cache."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd

from routing_ml.training import digest

ENCODER_ID = "BAAI/bge-small-en-v1.5"
ENCODER_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
DIMENSIONS = 384
CONTENT_TOKENS = 510
BATCH_SIZE = 16
ENCODER_FILES = ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json",
                 "special_tokens_map.json", "vocab.txt", "README.md", "1_Pooling/config.json")
ENCODER_CONFIG = dict(model_id=ENCODER_ID, revision=ENCODER_REVISION, dimensions=DIMENSIONS,
                      content_tokens_per_chunk=CONTENT_TOKENS, chunk_overlap=0,
                      pooling="L2 CLS per chunk; content-token-weighted mean; final L2",
                      instruction=None, dtype="float32", device="cpu", batch_size=BATCH_SIZE,
                      torch_threads=4, attention="eager", fine_tuned=False)


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def json_file(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def download_encoder(directory):
    """Explicit public-asset download; no credentials or hosted inference calls."""
    import requests
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    response = session.get(f"https://huggingface.co/api/models/{ENCODER_ID}/revision/{ENCODER_REVISION}",
                           params={"blobs": "true"}, timeout=60)
    response.raise_for_status()
    info = response.json()
    if info["sha"] != ENCODER_REVISION:
        raise ValueError("Encoder revision mismatch")
    remote = {row["rfilename"]: row for row in info["siblings"]}
    files = {}
    for name in ENCODER_FILES:
        path = directory / name
        expected = remote[name]
        expected_sha = expected.get("lfs", {}).get("sha256")
        if not path.exists() or (expected_sha and file_hash(path) != expected_sha):
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + ".part")
            print(f"Downloading pinned encoder asset: {name}", flush=True)
            with session.get(f"https://huggingface.co/{ENCODER_ID}/resolve/{ENCODER_REVISION}/{name}",
                             stream=True, timeout=60) as asset:
                asset.raise_for_status()
                with temporary.open("wb") as f:
                    for block in asset.iter_content(1024 * 1024):
                        f.write(block)
            temporary.replace(path)
        actual = file_hash(path)
        if expected_sha and actual != expected_sha:
            raise ValueError(f"Encoder asset checksum mismatch: {name}")
        if expected.get("size") is not None and path.stat().st_size != expected["size"]:
            raise ValueError(f"Encoder asset size mismatch: {name}")
        # Hugging Face's non-LFS blob ID is the SHA-1 of Git's blob header + bytes.
        if not expected_sha and expected.get("blobId"):
            data = path.read_bytes()
            blob = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
            if blob != expected["blobId"]:
                raise ValueError(f"Encoder Git blob checksum mismatch: {name}")
        files[name] = dict(sha256=actual, bytes=path.stat().st_size)
    json_file(directory / "manifest.json", dict(model_id=ENCODER_ID, revision=ENCODER_REVISION, files=files))
    verify_encoder(directory)


def verify_encoder(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["model_id"] != ENCODER_ID or manifest["revision"] != ENCODER_REVISION:
        raise ValueError("Incorrect encoder identity")
    if set(manifest["files"]) != set(ENCODER_FILES):
        raise ValueError("Incomplete encoder manifest")
    for name, expected in manifest["files"].items():
        if file_hash(directory / name) != expected["sha256"]:
            raise ValueError(f"Encoder asset hash mismatch: {name}")
    return manifest


def token_chunks(token_ids, size=CONTENT_TOKENS):
    if size < 1:
        raise ValueError("Positive chunk size required")
    return [token_ids[i:i + size] for i in range(0, len(token_ids), size)] or [[]]


def prompt_identity(prompts):
    if prompts.prompt_id.duplicated().any() or not prompts.prompt.map(lambda x: isinstance(x, str)).all():
        raise ValueError("Unique prompt IDs and text strings required")
    return digest(list(prompts[["prompt_id", "prompt"]].itertuples(index=False, name=None)))


def weighted_prompt_vectors(chunk_vectors, owners, weights, n_prompts):
    out = np.zeros((n_prompts, chunk_vectors.shape[1]), dtype=np.float64)
    np.add.at(out, owners, chunk_vectors.astype(np.float64) * np.asarray(weights)[:, None])
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    if not np.isfinite(out).all() or (norms == 0).any():
        raise ValueError("Invalid/empty prompt embeddings")
    return (out / norms).astype(np.float32)


def encode_prompts(prompts, encoder_dir, progress=print):
    """Only prompt_id and prompt are projected; every original token is retained."""
    import torch
    from transformers import AutoModel, AutoTokenizer
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    verify_encoder(encoder_dir)
    identity = prompt_identity(prompts)
    torch.manual_seed(3407)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    tokenizer = AutoTokenizer.from_pretrained(str(encoder_dir), local_files_only=True, trust_remote_code=False)
    model = AutoModel.from_pretrained(str(encoder_dir), local_files_only=True, trust_remote_code=False,
                                      use_safetensors=True, attn_implementation="eager").to("cpu").eval()
    if model.config.hidden_size != DIMENSIONS:
        raise ValueError("Encoder dimension mismatch")
    chunks, owners, weights, statistics = [], [], [], []
    for i, row in enumerate(prompts[["prompt_id", "prompt"]].itertuples(index=False)):
        ids = tokenizer(row.prompt, add_special_tokens=False, truncation=False, verbose=False)["input_ids"]
        pieces = token_chunks(ids)
        statistics.append(dict(prompt_id=row.prompt_id, content_tokens=len(ids), chunks=len(pieces)))
        for piece in pieces:
            chunks.append(tokenizer.build_inputs_with_special_tokens(piece))
            owners.append(i)
            weights.append(max(len(piece), 1))
    order = sorted(range(len(chunks)), key=lambda i: (len(chunks[i]), i))
    vectors = np.empty((len(chunks), DIMENSIONS), dtype=np.float32)
    start = time.perf_counter()
    with torch.inference_mode():
        for offset in range(0, len(order), BATCH_SIZE):
            indices = order[offset:offset + BATCH_SIZE]
            batch = tokenizer.pad({"input_ids": [chunks[i] for i in indices]}, padding=True, return_tensors="pt")
            result = model(**batch).last_hidden_state[:, 0]
            result = torch.nn.functional.normalize(result, p=2, dim=1)
            vectors[indices] = result.cpu().numpy()
            if offset % (BATCH_SIZE * 25) == 0 and progress:
                progress(f"Encoder: {min(offset+BATCH_SIZE, len(order))}/{len(order)} chunks; {time.perf_counter()-start:.1f}s", flush=True)
    result = weighted_prompt_vectors(vectors, owners, weights, len(prompts))
    return result, pd.DataFrame(statistics), dict(prompt_text_identity=identity, chunks=len(chunks),
                                                 content_tokens=sum(s["content_tokens"] for s in statistics),
                                                 inference_seconds=time.perf_counter() - start,
                                                 encoder_config=ENCODER_CONFIG)


def prepare_cache(data_dir, encoder_dir, cache_dir):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    prompts = pd.read_parquet(Path(data_dir) / "prompts.parquet", columns=["prompt_id", "prompt"]).sort_values("prompt_id").reset_index(drop=True)
    if (cache_dir / "manifest.json").exists():
        return load_cache(cache_dir, prompts)
    vectors, statistics, metadata = encode_prompts(prompts, encoder_dir)
    np.save(cache_dir / "vectors.npy", vectors, allow_pickle=False)
    statistics.to_parquet(cache_dir / "prompt_index.parquet", index=False)
    metadata["encoder_manifest_hash"] = file_hash(Path(encoder_dir) / "manifest.json")
    metadata["source_prompt_file_hash"] = file_hash(Path(data_dir) / "prompts.parquet")
    metadata["files"] = {name: file_hash(cache_dir / name) for name in ("vectors.npy", "prompt_index.parquet")}
    json_file(cache_dir / "manifest.json", metadata)
    return load_cache(cache_dir, prompts)


def load_cache(cache_dir, prompts=None):
    cache_dir = Path(cache_dir)
    manifest = json.loads((cache_dir / "manifest.json").read_text())
    if manifest["encoder_config"] != ENCODER_CONFIG:
        raise ValueError("Embedding configuration mismatch")
    for name, expected in manifest["files"].items():
        if file_hash(cache_dir / name) != expected:
            raise ValueError("Embedding cache hash mismatch")
    index = pd.read_parquet(cache_dir / "prompt_index.parquet")
    values = np.load(cache_dir / "vectors.npy", allow_pickle=False)
    if values.shape != (len(index), DIMENSIONS) or index.prompt_id.duplicated().any():
        raise ValueError("Embedding cache shape or ID mismatch")
    if not np.isfinite(values).all() or not np.allclose(np.linalg.norm(values, axis=1), 1, atol=1e-6):
        raise ValueError("Invalid embedding norms")
    if prompts is not None and manifest["prompt_text_identity"] != prompt_identity(prompts):
        raise ValueError("Embedding cache prompt text/ID mismatch")
    return pd.DataFrame(values, index=pd.Index(index.prompt_id, name="prompt_id")), manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="Download public pinned encoder files only")
    parser.add_argument("--encoder-dir", type=Path, default=Path("artifacts/router_v2/encoder"))
    parser.add_argument("--cache-dir", type=Path, default=Path("artifacts/router_v2/embeddings"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/llmrouterbench"))
    args = parser.parse_args()
    if args.download:
        download_encoder(args.encoder_dir)
    else:
        from contextlib import ExitStack
        from unittest.mock import patch
        def blocked(*args, **kwargs):
            raise RuntimeError("Encoder inference must be offline")
        with ExitStack() as stack:
            for target in ("socket.socket.connect", "socket.create_connection", "socket.getaddrinfo", "urllib.request.urlopen"):
                stack.enter_context(patch(target, side_effect=blocked))
            values, metadata = prepare_cache(args.data_dir, args.encoder_dir, args.cache_dir)
            print(f"Saved {values.shape} frozen local embeddings")


if __name__ == "__main__":
    main()
