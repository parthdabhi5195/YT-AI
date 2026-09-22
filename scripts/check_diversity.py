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

    Usage On Final Concatenated output: 
    python3 -c "
import random
random.seed(0); K=20000; res=[]
with open('data/clean/final/stories_text.jsonl') as f:
    for i, line in enumerate(f):
        if i < K: res.append(line)
        else:
            j = random.randint(0, i)
            if j < K: res[j] = line
open('/tmp/sample_20k.jsonl','w').writelines(res)
"
python3 scripts/check_diversity.py --input /tmp/sample_20k.jsonl --sample-size 2000

"""
import argparse
import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from pipeline_io import iter_jsonl  # noqa: E402

# imports to convert language into embeddings
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True) 
    parser.add_argument("--sample-size", type=int, default=500,
                         help="Cap the number of stories embedded, for speed")
    args = parser.parse_args()

    def bad_line(line_no, line, exc):
        print(f"  warning: line {line_no} is not valid JSON -- ignoring it")

    # read valid JSON rows one by one, skipping any that are not valid JSON or don't have a "story_text" field
    # Create a list of stories from valid rows
    # keeping this logic to avoid crashing in event of a mid-run failure 
    stories = [row["story_text"] for _, row in iter_jsonl(args.input, on_bad=bad_line)
               if isinstance(row, dict) and row.get("story_text")]

    # stories list is empty
    if not stories:
        raise SystemExit("No stories found in input file.")

    # If the number of stories exceeds the sample size, randomly sample a subset for embedding
    # We are randomly sampling to avoid bias since the stories are generated in batches and may have some ordering
    if len(stories) > args.sample_size:
        stories = random.sample(stories, args.sample_size)

    print(f"Embedding {len(stories)} stories...")
    model = SentenceTransformer("all-MiniLM-L6-v2") # sentence-transformer model to convert stories into embeddings
    embeddings = model.encode(stories, show_progress_bar=True)

    sims = cosine_similarity(embeddings) # normalized version of dot product between two vectors
    np.fill_diagonal(sims, np.nan) # remove self-similarity by setting diagonal to NaN (to avoid SKEWING the average)
    avg_sim = np.nanmean(sims) # average all the off-diagonal values
    max_pair = np.unravel_index(np.nanargmax(np.nan_to_num(sims, nan=-1)), sims.shape) # find pair of stories with highest similarity and return their index pair in the original stories list

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

"""
Small note on how cosine similarity works:

example:
#
# Vector 1 = [2, 1, 0]
# Vector 2 = [1, 1, 3]
# 
#               Vector 1 [2,1,0]      Vector 2 [1,1,3]
#             +-----------------+-----------------+
# Vector 1    |     1.0000      |     0.4045      |
# [2, 1, 0]   |  (Self-Match)   |  (Cross-Match)  |
#             +-----------------+-----------------+
# Vector 2    |     0.4045      |     1.0000      |
# [1, 1, 3]   |  (Cross-Match)  |  (Self-Match)   |
#             +-----------------+-----------------+
#
# In our case, the matrix is of size N x N, where N is the number of stories.
# The diagonal elements are self-matches. The off-diagonal elements are cosine similarities between different stories.
"""