from __future__ import annotations

import random
from pathlib import Path

from _shared.io import fact_text
from _shared.neural import batched, mean_pool, require_torch, save_checkpoint


class DualEncoderRetriever:
    def __init__(self, model_path: Path, device: str) -> None:
        torch, AutoModel, _, AutoTokenizer = require_torch()
        self.torch, self.device = torch, device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModel.from_pretrained(model_path, local_files_only=True).to(device)

    def _encode(self, texts: list[str], grad: bool):
        torch = self.torch
        tokens = self.tokenizer(texts, padding=True, truncation=True, max_length=128,
                                return_tensors="pt").to(self.device)
        context = torch.enable_grad() if grad else torch.no_grad()
        with context:
            output = self.model(**tokens).last_hidden_state
            embeddings = mean_pool(output, tokens["attention_mask"])
            return torch.nn.functional.normalize(embeddings, p=2, dim=1)

    def train_model(self, rows: list[dict], facts: list[dict], checkpoint: Path,
                    epochs: int, batch_size: int, seed: int, eviret: bool = False,
                    use_substitutes: bool | None = None,
                    use_distractors: bool | None = None,
                    use_hdm: bool | None = None) -> int:
        torch = self.torch
        use_substitutes = eviret if use_substitutes is None else use_substitutes
        use_distractors = eviret if use_distractors is None else use_distractors
        use_hdm = eviret if use_hdm is None else use_hdm
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-5)
        rng = random.Random(seed)
        universe = len(facts)
        difficulty_values = [float(row.get("hdm_difficulty", 0.0)) for row in rows]
        difficulty_min = min(difficulty_values, default=0.0)
        difficulty_max = max(difficulty_values, default=0.0)

        def hdm_weight(row: dict) -> float:
            normalized = ((float(row.get("hdm_difficulty", 0.0)) - difficulty_min) /
                          (difficulty_max - difficulty_min + 1e-12))
            return 1.0 + normalized
        optimizer_steps = 0
        for epoch in range(epochs):
            order = list(rows)
            rng.shuffle(order)
            self.model.train()
            total = 0.0
            for batch in batched(order, batch_size):
                questions, positives, positives_2, negatives, negatives_2, weights = [], [], [], [], [], []
                for row in batch:
                    groups = row.get("substitute_groups", {})
                    if use_substitutes and groups:
                        anchor = rng.choice(list(groups))
                        members = [int(anchor), *groups[anchor]]
                        positive = rng.choice(members)
                        positive_2 = rng.choice(members)
                    else:
                        positive = rng.choice(row["anchors"])
                        positive_2 = positive
                    blocked = set(row["anchors"]) | set(row["substitutes"])
                    hard_pool = row["distractors"] if use_distractors else []
                    negative = rng.choice(hard_pool) if hard_pool else rng.randrange(universe)
                    while negative in blocked:
                        negative = rng.randrange(universe)
                    random_negative = rng.randrange(universe)
                    while random_negative in blocked:
                        random_negative = rng.randrange(universe)
                    questions.append(row["question"])
                    positives.append(fact_text(facts[positive]))
                    positives_2.append(fact_text(facts[positive_2]))
                    negatives.append(fact_text(facts[negative]))
                    negatives_2.append(fact_text(facts[random_negative]))
                    weights.append(hdm_weight(row) if use_hdm else 1.0)
                q = self._encode(questions, True)
                p = self._encode(positives, True)
                n = self._encode(negatives, True)
                base = q @ p.T
                if use_substitutes:
                    p2 = self._encode(positives_2, True)
                    n2 = self._encode(negatives_2, True)
                    own_positive = (q * p2).sum(1, keepdim=True)
                    logits = torch.cat((base, own_positive,
                                        (q * n).sum(1, keepdim=True),
                                        (q * n2).sum(1, keepdim=True)), dim=1) / 0.05
                    diagonal = base.diagonal() / 0.05
                    positive_logits = torch.stack((diagonal, own_positive.view(-1) / 0.05), dim=1)
                    losses = torch.logsumexp(logits, dim=1) - torch.logsumexp(positive_logits, dim=1)
                else:
                    logits = torch.cat((base, (q * n).sum(1, keepdim=True)), dim=1) / 0.05
                    targets = torch.arange(len(batch), device=self.device)
                    losses = torch.nn.functional.cross_entropy(logits, targets, reduction="none")
                weight_tensor = torch.tensor(weights, device=self.device)
                loss = (losses * weight_tensor).sum() / weight_tensor.sum()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                optimizer_steps += 1
                total += float(loss.detach())
            print(f"[train] epoch={epoch + 1}/{epochs} batches={(len(order)+batch_size-1)//batch_size} loss_sum={total:.6f}", flush=True)
        save_checkpoint(self.model, self.tokenizer, checkpoint,
                        {"epochs": epochs, "examples": len(rows), "seed": seed,
                         "optimizer_steps": optimizer_steps,
                         "objective": "group_multi_positive" if use_substitutes else "anchor_inbatch",
                         "use_substitutes": use_substitutes,
                         "use_distractors": use_distractors,
                         "use_hdm": use_hdm,
                         "hdm_weight": "1+(D-D_train_min)/(D_train_max-D_train_min+epsilon)" if use_hdm else None})
        return optimizer_steps

    def encode_corpus(self, facts: list[dict], indices: list[int], batch_size: int = 256):
        self.model.eval()
        output = {}
        for chunk in batched(indices, batch_size):
            emb = self._encode([fact_text(facts[index]) for index in chunk], False).cpu()
            output.update({index: emb[pos] for pos, index in enumerate(chunk)})
        return output

    def score(self, question: str, candidates: list[int], cache: dict[int, object]) -> dict[int, float]:
        q = self._encode([question], False).cpu()[0]
        matrix = self.torch.stack([cache[index] for index in candidates])
        values = matrix @ q
        return {index: float(values[position]) for position, index in enumerate(candidates)}
