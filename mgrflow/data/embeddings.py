"""Build a fixed ESM-2 marker table from a JSON mapping of keys to sequences."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch


@torch.no_grad()
def embed_sequence(model, tokenizer, sequence: str, device: torch.device) -> torch.Tensor:
    sequence = "".join(sequence.split()).upper()
    if not sequence:
        raise ValueError("Every supplied protein sequence must contain residues.")
    total = torch.zeros(640, device=device)
    for start in range(0, len(sequence), 1022):
        chunk = sequence[start : start + 1022]
        tokens = tokenizer(chunk, return_tensors="pt", return_special_tokens_mask=True).to(device)
        special = tokens.pop("special_tokens_mask").bool()
        hidden = model(**tokens).last_hidden_state
        residues = tokens["attention_mask"].bool() & ~special
        total += hidden[residues].float().mean(dim=0) * len(chunk)
    return (total / len(sequence)).cpu()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sequences", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model", default="biohub/esm2_t30_150M_UR50D")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    from transformers import AutoTokenizer, EsmModel

    device = torch.device(args.device)
    sequences = json.loads(args.sequences.read_text())
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = EsmModel.from_pretrained(args.model, add_pooling_layer=False).eval().to(device)
    keys = sorted(sequences)
    vectors = []
    for key in keys:
        entries = sequences[key]
        entries = [entries] if isinstance(entries, str) else entries
        vectors.append(
            torch.stack(
                [embed_sequence(model, tokenizer, sequence, device) for sequence in entries]
            ).mean(dim=0)
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as file:
        np.savez(file, marker_keys=np.array(keys), embeddings=torch.stack(vectors).numpy())


if __name__ == "__main__":
    main()
