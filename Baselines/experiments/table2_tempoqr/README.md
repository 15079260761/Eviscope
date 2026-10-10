# TempoQR paired Clean/Full experiment

The formal matrix implements all three Table-2 configurations:

1. Soft supervision with the CronKGQA model (`soft_cronkgqa`)
2. Soft supervision with the TempoQR model (`soft_tempoqr`)
3. Hard supervision with the TempoQR model (`hard_tempoqr`)

Each configuration trains once on the same deterministic 10,000-question
training cohort, selects its checkpoint on the same 2,000-question development
cohort, and evaluates the same 1,691 Test questions under Clean and Full pools.
Clean and Full share one checkpoint within each configuration. Each candidate
pool has exactly 1,000 facts. Test labels are unavailable to model code and are
read only by the official evaluator.

`launch_first_two.sh` starts configurations 1 and 2 concurrently on GPUs 0 and
1. `launch_third.sh GPU` starts configuration 3 later on whichever GPU is free.
All output is unbuffered and remains visible in tmux; no log redirection is used.
