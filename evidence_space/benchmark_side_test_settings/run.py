#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, os, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; EXP2 = HERE.parent; EXP1 = EXP2.parent / "main_benchmark"
sys.path[:0] = [str(EXP2), str(EXP1)]
from protocol import (CONDITIONS, FrozenHashEncoder, dataset_spec, load_facts, load_prepared,
                      labels_for_condition, make_retriever, save_and_evaluate,
                      score_union, top10, write_json)
def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("--method",choices=("bm25","dual_encoder","cross_encoder","eviret"),required=True); p.add_argument("--dataset",choices=("cron","icews"),required=True); p.add_argument("--exp1-run-id",default=os.environ.get("EXP1_RUN_ID","exp1_full_v1")); p.add_argument("--run-id",default=os.environ.get("RUN_ID","exp2_v2")); p.add_argument("--device",default="cuda:0"); p.add_argument("--score-batch-size",type=int,default=128); p.add_argument("--resume-verified",action="store_true"); a=p.parse_args()
    root=HERE/"results"/a.run_id/a.method/a.dataset
    if root.exists():
        manifest_path=root/"run_manifest.json"; summary_path=root/"summary.json"
        complete=(manifest_path.is_file() and summary_path.is_file() and
                  all((root/c/"predictions.jsonl").is_file() and
                      (root/c/"metrics_k1.json").is_file() and
                      (root/c/"metrics_k10.json").is_file() for c in CONDITIONS))
        if a.resume_verified and complete:
            manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
            if (manifest.get("protocol_version")=="exp2-section-2.1-2.6-v2" and
                    manifest.get("method")==a.method and manifest.get("dataset")==a.dataset):
                print(f"[SKIP VERIFIED] benchmark-side {a.method}/{a.dataset}",flush=True); return
        raise FileExistsError(f"unverified or incomplete benchmark result directory: {root}")
    spec=dataset_spec(a.dataset); facts=load_facts(spec.facts); questions,labels,_,pools=load_prepared(a.dataset)
    checkpoint=EXP1/"checkpoints"/a.exp1_run_id/a.method/a.dataset; source=None if a.method=="bm25" else checkpoint
    if source is not None and not (source/"config.json").is_file(): raise FileNotFoundError(f"missing frozen Experiment 1 checkpoint: {source}")
    retriever=make_retriever(a.method,source,facts,a.device); qmap={r["question_id"]:r["question"] for r in questions}
    indices=sorted({i for mode in ("substitute","full") for values in pools[mode].values() for i in values})
    cache=retriever.encode_corpus(facts,indices) if a.method in ("dual_encoder","eviret") else None
    predictions={c:[] for c in CONDITIONS}
    for pos,qid in enumerate(sorted(qmap),1):
        union=sorted(set(pools["substitute"][qid])|set(pools["full"][qid])); scores=score_union(a.method,retriever,qmap[qid],union,facts,cache,a.score_batch_size)
        no_dis=top10(scores,pools["substitute"][qid]); with_dis=top10(scores,pools["full"][qid])
        for c in ("original","sub"): predictions[c].append({"question_id":qid,"ranked_fact_indices":no_dis})
        for c in ("dis","full"): predictions[c].append({"question_id":qid,"ranked_fact_indices":with_dis})
        if pos%100==0 or pos==len(qmap): print(f"[benchmark-side] {a.method}/{a.dataset} {pos}/{len(qmap)}",flush=True)
    encoder=FrozenHashEncoder(spec.facts); summary={}
    for c in CONDITIONS: summary[c]=save_and_evaluate(root/c,predictions[c],labels_for_condition(labels,c),facts,encoder,spec.bs_tau)
    if predictions["original"] != predictions["sub"] or predictions["dis"] != predictions["full"]:
        raise RuntimeError("factorial ranking identity was violated")
    test_ids = "\n".join(sorted(qmap)) + "\n"
    write_json(root/"summary.json",summary); write_json(root/"run_manifest.json",{
        "protocol_version":"exp2-section-2.1-2.6-v2",
        "direction":"benchmark_side","method":a.method,"dataset":a.dataset,
        "frozen_checkpoint":str(source) if source else None,
        "test_questions":len(qmap),
        "test_question_ids_sha256":hashlib.sha256(test_ids.encode("utf-8")).hexdigest(),
        "candidate_protocol":{"distractor_off":"substitute_pool","distractor_on":"full_pool"},
        "ranking_identity":["original=sub","dis=full"],
        "metric_semantics":{
            "hits":"substitute validity changes for +Sub/Full",
            "bs_cov_nbs":"paper definitions match retrieved facts only to anchors; substitute-label toggling cannot change these metrics when ranking is fixed"
        }
    })
    print(f"[SUCCESS] benchmark-side {a.method}/{a.dataset}",flush=True)
if __name__=="__main__": main()
