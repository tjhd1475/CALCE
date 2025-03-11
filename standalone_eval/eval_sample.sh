#!/usr/bin/env bash
# Usage: bash standalone_eval/eval_sample.sh
# submission_path=standalone_eval/sample_val_preds.jsonl
submission_path=/data2/xiepeiyu/save/mrhd_BLIP/QVH/QVH_CLUSTER_CAP_FILTER_EVAL/result/hl_val_submission.jsonl
gt_path=/data2/xiepeiyu/qvhighlights/annotations/highlight_val_release.jsonl
save_path=standalone_eval/val_preds_metrics.json

PYTHONPATH=$PYTHONPATH:. python standalone_eval/eval.py \
--submission_path ${submission_path} \
--gt_path ${gt_path} \
--save_path ${save_path}
