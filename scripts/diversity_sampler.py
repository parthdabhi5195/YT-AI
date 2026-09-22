"""
Samples a random combination of story ingredients (setting, occupation,
relationship dynamic, incident seed, two character names) from the pools
in config/diversity_pools.json.

Import and call sample_combo() from any generation script. Run this file
directly to see example output.
"""
import json
import random
from pathlib import Path
from typing import Optional

POOLS_PATH = Path(__file__).parent.parent / "config" / "diversity_pools.json"


def load_pools(path: Path = POOLS_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class DiversitySampler:
    """
    Each pool is shuffled once and then drawn from in order, reshuffling
    only once fully exhausted. This is deliberately NOT plain
    random.choice() on every call -- with millions of draws, pure random
    choice clusters more than people expect (the birthday paradox), so
    cycling through a shuffled pool guarantees every item gets used before
    anything repeats.
    """

    def __init__(self, pools: Optional[dict] = None, seed: Optional[int] = None):
        self.pools = pools or load_pools()
        self.rng = random.Random(seed)
        self._cycles = {key: self._new_cycle(key) for key in self.pools}

    def _new_cycle(self, key: str):
        items = list(self.pools[key])
        self.rng.shuffle(items)
        return iter(items)

    def _draw(self, key: str):
        try:
            return next(self._cycles[key])
        except StopIteration:
            self._cycles[key] = self._new_cycle(key)
            return next(self._cycles[key])

    def sample_combo(self, num_names: int = 2) -> dict:
        """
        num_names: how many DISTINCT character names to draw. Generation
        callers pass one per role in the template (see prompts.names_needed)
        so the model is handed a name for every character. Left to invent
        supporting names itself, it collapses onto the same few defaults --
        in a 100-story pilot, Sarah appeared in 22% of stories and Mark 21%.

        With the default of 2 this draws exactly the same stream as before,
        so seeds stay reproducible for existing callers.
        """
        num_names = min(max(2, num_names), len(self.pools["names"]))
        names = []
        # A repeat is only possible at a cycle boundary, when the reshuffled
        # pool starts with a name drawn at the end of the previous cycle.
        # Skipping it and drawing again always terminates: a fresh cycle
        # holds every name exactly once.
        while len(names) < num_names:
            name = self._draw("names")
            if name not in names:
                names.append(name)
        return {
            "setting": self._draw("settings"),
            "occupation": self._draw("occupations"),
            "relationship_dynamic": self._draw("relationship_dynamics"),
            "incident_seed": self._draw("incident_seeds"),
            "hook_style": self._draw("hook_styles"),
            "character_names": names,
        }


if __name__ == "__main__":
    sampler = DiversitySampler(seed=42)
    print("Three example combos:\n")
    for i in range(3):
        print(f"--- combo {i+1} ---")
        print(json.dumps(sampler.sample_combo(), indent=2))
        print()
