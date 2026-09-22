"""
Quick diversity / mode-collapse check on a batch of generated stories.

Embeds every story with a small sentence-transformer model and reports the
average pairwise cosine similarity across a sample.

Rough rule of thumb:
    < 0.5    healthy diversity
    0.5-0.7  some repetition -- worth reading a sample
    > 0.7    likely mode collapse -- widen config/diversity_pools.json
             before scaling up

Usage:
    pip install sentence-transformers scikit-learn numpy
    python scripts/check_diversity.py --input data/pilot_output/pilot.jsonl
"""
import argparse
import json
import random

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--sample-size", type=int, default=500,
                         help="Cap the number of stories embedded, for speed")
    args = parser.parse_args()

    stories = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            stories.append(json.loads(line)["story_text"])

    if not stories:
        raise SystemExit("No stories found in input file.")

    if len(stories) > args.sample_size:
        stories = random.sample(stories, args.sample_size)

    print(f"Embedding {len(stories)} stories...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(stories, show_progress_bar=True)

    sims = cosine_similarity(embeddings)
    np.fill_diagonal(sims, np.nan)
    avg_sim = np.nanmean(sims)
    max_pair = np.unravel_index(np.nanargmax(np.nan_to_num(sims, nan=-1)), sims.shape)

    print(f"\nAverage pairwise cosine similarity: {avg_sim:.3f}")
    if avg_sim > 0.7:
        print("WARNING: this suggests mode collapse. Widen your diversity "
              "pools (config/diversity_pools.json) before scaling up.")
    elif avg_sim > 0.5:
        print("Some repetition present -- worth reading a sample before scaling.")
    else:
        print("Looks healthy -- diversity pools are doing their job.")

    print(f"\nMost similar pair found (indices {max_pair}):")
    print("---")
    print(stories[max_pair[0]][:300])
    print("---")
    print(stories[max_pair[1]][:300])


if __name__ == "__main__":
    main()
