import numpy as np
from collections import OrderedDict, defaultdict
import json
import time
import copy
import multiprocessing as mp
from lavis.tasks.mr_utils import (
    compute_average_precision_detection,
    compute_temporal_iou_batch_cross,
    compute_temporal_iou_batch_paired,
    load_jsonl,
    get_ap,
)

from lavis.tasks.metrics_own import r1_and_mIoU


def compute_average_precision_detection_wrapper(
    input_triple, tiou_thresholds=np.linspace(0.5, 0.95, 10)
):
    qid, ground_truth, prediction = input_triple
    scores = compute_average_precision_detection(
        ground_truth, prediction, tiou_thresholds=tiou_thresholds
    )
    return qid, scores


def compute_mr_ap(
    submission,
    ground_truth,
    iou_thds=np.linspace(0.5, 0.95, 10),
    max_gt_windows=None,
    max_pred_windows=None,
    num_workers=8,
    chunksize=50,
):
    iou_thds = [float(f"{e:.2f}") for e in iou_thds]
    pred_qid2data = defaultdict(list)
    gt_qid2data = defaultdict(list)
    for d in submission:
        qid = d["qid"]

        # get predicted windows
        pred_windows = (
            d["pred_relevant_windows"][:max_pred_windows]
            if max_pred_windows is not None
            else d["pred_relevant_windows"]
        )
        for w in pred_windows:
            pred_qid2data[qid].append(
                {
                    "video-id": d["qid"],  # in order to use the API
                    "t-start": w[0],
                    "t-end": w[1],
                }
            )

        # get target windows
        gt_windows = (
            d["relevant_windows"][:max_gt_windows]
            if max_gt_windows is not None
            else d["relevant_windows"]
        )
        for w in gt_windows:
            gt_qid2data[qid].append(
                {"video-id": d["qid"], "t-start": w[0], "t-end": w[1]}
            )

    qid2ap_list = {}
    # start_time = time.time()
    data_triples = [
        [qid, gt_qid2data[qid], pred_qid2data[qid]] for qid in pred_qid2data
    ]
    from functools import partial

    compute_ap_from_triple = partial(
        compute_average_precision_detection_wrapper, tiou_thresholds=iou_thds
    )

    if num_workers > 1:
        with mp.Pool(num_workers) as pool:
            for qid, scores in pool.imap_unordered(
                compute_ap_from_triple, data_triples, chunksize=chunksize
            ):
                qid2ap_list[qid] = scores
    else:
        for data_triple in data_triples:
            qid, scores = compute_ap_from_triple(data_triple)
            qid2ap_list[qid] = scores

    # print(f"compute_average_precision_detection {time.time() - start_time:.2f} seconds.")
    ap_array = np.array(list(qid2ap_list.values()))  # (#queries, #thd)
    ap_thds = ap_array.mean(0)  # mAP at different IoU thresholds.
    iou_thd2ap = dict(zip([str(e) for e in iou_thds], ap_thds))
    iou_thd2ap["average"] = np.mean(ap_thds)
    # formatting
    iou_thd2ap = {k: float(f"{100 * v:.2f}") for k, v in iou_thd2ap.items()}
    return iou_thd2ap


def compute_mr_r1(submission, ground_truth, iou_thds=np.linspace(0.5, 0.95, 10)):
    """If a predicted segment has IoU >= iou_thd with one of the 1st GT segment, we define it positive"""
    iou_thds = [float(f"{e:.2f}") for e in iou_thds]
    pred_qid2window = {
        d["qid"]: d["pred_relevant_windows"][0][:2] for d in submission
    }  # :2 rm scores
    # gt_qid2window = {d["qid"]: d["relevant_windows"][0] for d in ground_truth}
    gt_qid2window = {}
    for d in ground_truth:
        cur_gt_windows = d["relevant_windows"]
        cur_qid = d["qid"]
        cur_max_iou_idx = 0
        if len(cur_gt_windows) > 0:  # select the GT window that has the highest IoU
            cur_ious = compute_temporal_iou_batch_cross(
                np.array([pred_qid2window[cur_qid]]), np.array(d["relevant_windows"])
            )[0]
            cur_max_iou_idx = np.argmax(cur_ious)
        gt_qid2window[cur_qid] = cur_gt_windows[cur_max_iou_idx]

    qids = list(pred_qid2window.keys())
    pred_windows = np.array([pred_qid2window[k] for k in qids]).astype(float)
    gt_windows = np.array([gt_qid2window[k] for k in qids]).astype(float)
    pred_gt_iou = compute_temporal_iou_batch_paired(pred_windows, gt_windows)
    iou_thd2recall_at_one = {}
    for thd in iou_thds:
        iou_thd2recall_at_one[str(thd)] = float(
            f"{np.mean(pred_gt_iou >= thd) * 100:.2f}"
        )

    # number of invalid predictions
    invalid_pred_num = 0
    for pred in pred_windows:
        if -1 in pred:
            invalid_pred_num += 1

    # avg recall@1
    r1_avg = np.mean(list(iou_thd2recall_at_one.values()))

    # mean IoU
    mIoU = np.mean(pred_gt_iou)

    return iou_thd2recall_at_one, r1_avg, mIoU, invalid_pred_num


def get_window_len(window):
    return window[1] - window[0]


def get_data_by_range(submission, ground_truth, len_range):
    """keep queries with ground truth window length in the specified length range.
    Args:
        submission:
        ground_truth:
        len_range: [min_l (int), max_l (int)]. the range is (min_l, max_l], i.e., min_l < l <= max_l
    """
    min_l, max_l = len_range
    if min_l == 0 and max_l == 150:  # min and max l in dataset
        return submission, ground_truth

    # only keep ground truth with windows in the specified length range
    # if multiple GT windows exists, we only keep the ones in the range
    ground_truth_in_range = []
    gt_qids_in_range = set()
    for d in ground_truth:
        rel_windows_in_range = [
            w for w in d["relevant_windows"] if min_l < get_window_len(w) <= max_l
        ]
        if len(rel_windows_in_range) > 0:
            d = copy.deepcopy(d)
            d["relevant_windows"] = rel_windows_in_range
            ground_truth_in_range.append(d)
            gt_qids_in_range.add(d["qid"])

    # keep only submissions for ground_truth_in_range
    submission_in_range = []
    for d in submission:
        if d["qid"] in gt_qids_in_range:
            submission_in_range.append(copy.deepcopy(d))

    return submission_in_range, ground_truth_in_range


def eval_moment_retrieval(submission, ground_truth, verbose=True):

    # To use API without much modification, keep names as they are.
    # They simply won't mean anything anymore since we removed iterating over different ranges
    # that were specific to the QVH dataset
    range_names = ["short", "middle", "long", "full"]

    ret_metrics = {}
    for name in range_names:
        if verbose:
            start_time = time.time()

        _submission = submission
        _ground_truth = ground_truth

        print(
            f"{name}: {len(_ground_truth)}/{len(ground_truth)}="
            f"{100*len(_ground_truth)/len(ground_truth):.2f} examples."
        )
        iou_thd2average_precision = compute_mr_ap(
            _submission, _ground_truth, num_workers=8, chunksize=50
        )
        iou_thd2recall_at_one, r1_avg, mIoU, invalid_pred_num = compute_mr_r1(
            _submission, _ground_truth
        )
        # iou_thd2recall_at_one, r1_avg, mIoU, invalid_pred_num = r1_and_mIoU(_submission)
        ret_metrics[name] = {
            "mAP": iou_thd2average_precision,
            "R1": iou_thd2recall_at_one,
            "R1-avg": r1_avg,
            "mIoU": mIoU,
            "invalid_pred_num": invalid_pred_num,
        }
        if verbose:
            print(
                f"[eval_moment_retrieval] [{name}] {time.time() - start_time:.2f} seconds"
            )
    return ret_metrics


def compute_hl_hit1(qid2preds, qid2gt_scores_binary):
    qid2max_scored_clip_idx = {
        k: np.argmax(v["pred_saliency_scores"]) for k, v in qid2preds.items()
    }
    hit_scores = np.zeros((len(qid2preds), 3))
    qids = list(qid2preds.keys())
    for idx, qid in enumerate(qids):
        pred_clip_idx = qid2max_scored_clip_idx[qid]
        gt_scores_binary = qid2gt_scores_binary[qid]  # (#clips, 3)
        if pred_clip_idx < len(gt_scores_binary):
            hit_scores[idx] = gt_scores_binary[pred_clip_idx]
    # aggregate scores from 3 separate annotations (3 workers) by taking the max.
    # then average scores from all queries.
    hit_at_one = float(f"{100 * np.mean(np.max(hit_scores, 1)):.2f}")
    return hit_at_one


def compute_hl_ap(qid2preds, qid2gt_scores_binary, num_workers=8, chunksize=50):
    qid2pred_scores = {k: v["pred_saliency_scores"] for k, v in qid2preds.items()}
    ap_scores = np.zeros((len(qid2preds), 3))  # (#preds, 3)
    qids = list(qid2preds.keys())
    input_tuples = []
    for idx, qid in enumerate(qids):
        for w_idx in range(3):  # annotation score idx
            y_true = qid2gt_scores_binary[qid][:, w_idx]
            y_predict = np.array(qid2pred_scores[qid])
            input_tuples.append((idx, w_idx, y_true, y_predict))

    if num_workers > 1:
        with mp.Pool(num_workers) as pool:
            for idx, w_idx, score in pool.imap_unordered(
                compute_ap_from_tuple, input_tuples, chunksize=chunksize
            ):
                ap_scores[idx, w_idx] = score
    else:
        for input_tuple in input_tuples:
            idx, w_idx, score = compute_ap_from_tuple(input_tuple)
            ap_scores[idx, w_idx] = score

    # it's the same if we first average across different annotations, then average across queries
    # since all queries have the same #annotations.
    mean_ap = float(f"{100 * np.mean(ap_scores):.2f}")
    return mean_ap


def compute_ap_from_tuple(input_tuple):
    idx, w_idx, y_true, y_predict = input_tuple
    if len(y_true) < len(y_predict):
        # print(f"len(y_true) < len(y_predict) {len(y_true), len(y_predict)}")
        y_predict = y_predict[: len(y_true)]
    elif len(y_true) > len(y_predict):
        # print(f"len(y_true) > len(y_predict) {len(y_true), len(y_predict)}")
        _y_predict = np.zeros(len(y_true))
        _y_predict[: len(y_predict)] = y_predict
        y_predict = _y_predict

    score = get_ap(y_true, y_predict)
    return idx, w_idx, score


def eval_submission(submission, ground_truth=None, verbose=True, match_number=True):
    """
    Args:
        submission: list(dict), each dict is {
            qid: str,
            query: str,
            vid: str,
            pred_relevant_windows: list([st, ed]),
            pred_saliency_scores: list(float), len == #clips in video.
                i.e., each clip in the video will have a saliency score.
        }
        ground_truth: list(dict), each dict is     {
          "qid": 7803,
          "query": "Man in gray top walks from outside to inside.",
          "duration": 150,
          "vid": "RoripwjYFp8_360.0_510.0",
          "relevant_clip_ids": [13, 14, 15, 16, 17]
          "saliency_scores": [[4, 4, 2], [3, 4, 2], [2, 2, 3], [2, 2, 2], [0, 1, 3]]
               each sublist corresponds to one clip in relevant_clip_ids.
               The 3 elements in the sublist are scores from 3 different workers. The
               scores are in [0, 1, 2, 3, 4], meaning [Very Bad, ..., Good, Very Good]
        }
        verbose:
        match_number:

    Returns:

    """
    if ground_truth is None:
        ground_truth = submission
    pred_qids = set([e["qid"] for e in submission])
    gt_qids = set([e["qid"] for e in ground_truth])
    if match_number:
        assert pred_qids == gt_qids, (
            f"qids in ground_truth and submission must match. "
            f"use `match_number=False` if you wish to disable this check"
        )
    else:  # only leave the items that exists in both submission and ground_truth
        shared_qids = pred_qids.intersection(gt_qids)
        submission = [e for e in submission if e["qid"] in shared_qids]
        ground_truth = [e for e in ground_truth if e["qid"] in shared_qids]

    eval_metrics = {}
    eval_metrics_brief = OrderedDict()
    if "pred_relevant_windows" in submission[0]:
        moment_ret_scores = eval_moment_retrieval(
            submission, submission, verbose=verbose
        )
        eval_metrics.update({'mr':moment_ret_scores})
        moment_ret_scores_brief = {
            "MR-full-mAP": moment_ret_scores["full"]["mAP"]["average"],
            "MR-full-mAP@0.5": moment_ret_scores["full"]["mAP"]["0.5"],
            "MR-full-mAP@0.75": moment_ret_scores["full"]["mAP"]["0.75"],
            "MR-short-mAP": moment_ret_scores["short"]["mAP"]["average"],
            "MR-middle-mAP": moment_ret_scores["middle"]["mAP"]["average"],
            "MR-long-mAP": moment_ret_scores["long"]["mAP"]["average"],
            "MR-full-R1@0.5": moment_ret_scores["full"]["R1"]["0.5"],
            "MR-full-R1@0.7": moment_ret_scores["full"]["R1"]["0.7"],
            "MR-full-R1-avg": moment_ret_scores["full"]["R1-avg"],
            "MR-full-mIoU": moment_ret_scores["full"]["mIoU"],
            "MR-full-invalid_pred_num": moment_ret_scores["full"]["invalid_pred_num"],
        }
        eval_metrics_brief.update(
            sorted(
                [(k, v) for k, v in moment_ret_scores_brief.items()], key=lambda x: x[0]
            )
        )



    # sort by keys
    final_eval_metrics = OrderedDict()
    final_eval_metrics["brief"] = eval_metrics_brief
    final_eval_metrics.update(
        sorted([(k, v) for k, v in eval_metrics.items()], key=lambda x: x[0])
    )
    return final_eval_metrics


def eval_main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Moments and Highlights Evaluation Script"
    )
    parser.add_argument(
        "--submission_path", type=str, help="path to generated prediction file"
    )
    parser.add_argument("--gt_path", type=str, help="path to GT file")
    parser.add_argument("--save_path", type=str, help="path to save the results")
    parser.add_argument("--not_verbose", action="store_true")
    args = parser.parse_args()

    verbose = not args.not_verbose
    submission = load_jsonl(args.submission_path)
    gt = load_jsonl(args.gt_path)
    results = eval_submission(submission, gt, verbose=verbose)
    if verbose:
        print(json.dumps(results, indent=4))

    with open(args.save_path, "w") as f:
        f.write(json.dumps(results, indent=4))


if __name__ == "__main__":
    eval_main()
