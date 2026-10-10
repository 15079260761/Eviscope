from __future__ import annotations

import random
from pathlib import Path

from _shared.io import fact_text
from _shared.neural import batched, require_torch, save_checkpoint


class CrossEncoderRetriever:
    def __init__(self, model_path: Path, device: str) -> None:
        torch, _, AutoModelForSequenceClassification, AutoTokenizer = require_torch()
        self.torch, self.device = torch, device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path, local_files_only=True, ignore_mismatched_sizes=True,
            num_labels=1).to(device)

    def _logits(self, questions: list[str], facts: list[str], grad: bool):
        torch = self.torch
        tokens = self.tokenizer(questions, facts, padding=True, truncation=True,
                                max_length=192, return_tensors="pt").to(self.device)
        context = torch.enable_grad() if grad else torch.no_grad()
        with context:
            return self.model(**tokens).logits.view(-1)

    def train_model(self, rows: list[dict], facts: list[dict], checkpoint: Path,
                    epochs: int, batch_size: int, seed: int,
                    use_substitutes: bool = False,
                    use_distractors: bool = False) -> None:
        torch, rng = self.torch, random.Random(seed)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-5)
        universe = len(facts)
        for epoch in range(epochs):
            pairs = []
            for row in rows:
                positive_pool = list(row["anchors"])
                if use_substitutes:
                    positive_pool.extend(row["substitutes"])
                positive = rng.choice(positive_pool)
                pairs.append((row["question"], fact_text(facts[positive]), 1.0))
                blocked = set(row["anchors"]) | set(row["substitutes"])
                for _ in range(3):
                    negative = (rng.choice(row["distractors"])
                                if use_distractors and row["distractors"]
                                else rng.randrange(universe))
                    while negative in blocked:
                        negative = rng.randrange(universe)
                    pairs.append((row["question"], fact_text(facts[negative]), 0.0))
            rng.shuffle(pairs)
            self.model.train()
            total = 0.0
            for batch in batched(pairs, batch_size):
                logits = self._logits([x[0] for x in batch], [x[1] for x in batch], True)
                labels = torch.tensor([x[2] for x in batch], device=self.device)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                total += float(loss.detach())
            print(f"[train] epoch={epoch + 1}/{epochs} pairs={len(pairs)} loss_sum={total:.6f}", flush=True)
        save_checkpoint(self.model, self.tokenizer, checkpoint,
                        {"epochs": epochs, "questions": len(rows), "pairs_per_question": 4,
                         "seed": seed, "objective": "pointwise_bce",
                         "use_substitutes": use_substitutes,
                         "use_distractors": use_distractors})

    def score(self, question: str, candidates: list[int], facts: list[dict], batch_size: int = 128):
        self.model.eval()
        result = {}
        for chunk in batched(candidates, batch_size):
            logits = self._logits([question] * len(chunk),
                                  [fact_text(facts[index]) for index in chunk], False).cpu()
            result.update({index: float(logits[pos]) for pos, index in enumerate(chunk)})
        return result
