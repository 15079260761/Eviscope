# MultiQA and RTQA — paired EviScope CronQuestions experiment

- Train: MultiQA once on the deterministic 10,000-row Train subset; validate
  on 2,000 Dev rows. Clean and Full reuse that exact checkpoint.
- RTQA: inference-only BGE-M3 retrieval component from the released RTQA
  pipeline. The external answer-generation LLM is intentionally excluded
  because EviScope evaluates ranked evidence and the release provides no fixed
  LLM/API artifact.
- Test: the same 1,691 Full-eligible questions in both conditions.
- Candidate space: exactly 1,000 facts/question. Clean exposes Anchor+Neutral;
  Full exposes Anchor+Substitute+Distractor+Neutral.
- Leakage: Test answers are absent from model input. Private evidence roles are
  used only by trusted pool materialization and the official evaluator.
