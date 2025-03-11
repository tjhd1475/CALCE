import json
import ast
import re

def save_json(content, save_path):
    with open(save_path, 'w') as f:
        f.write(json.dumps(content))
def load_jsonl(filename):
    with open(filename, "r") as f:
        return [json.loads(l.strip("\n")) for l in f.readlines()]
def load_json(filename):
    with open(filename, "r") as f:
        return json.load(f)
def save_jsonl(content, save_path):
    with open(save_path, 'w') as f:
        for l in content:
            f.write(json.dumps(l) + "\n")

def moment_str_to_list(m):
    """Convert a string of moments to a list of moments.
    If predicted string is not a list, it means that the model has not yet learned to predict the right format.
    In that case, we return [[-1, -1]] to represent an error.
    This will then lead to an IoU of 0.
    Args:
        m (str): a string of moments, e.g. "[[0, 1], [4, 7]]"
    Returns:
        list: a list of moments, e.g. [[0, 1], [4, 7]]
    """
    if m == "[[-1, -1]]":
        return [[-1, -1]]

    # check if the string has the right format of a nested list using regex
    # the list should look like this: [[0, 1], [4, 7], ...]
    # if not, return [[-1, -1]]
    if not re.match(r"\[\[.*\]\]", m):
        return [[-1, -1]]

    try:
        _m = ast.literal_eval(m)
    except:
        return [[-1, -1]]

    # if _m is not a list, it means that the model has not predicted any relevant windows
    # return error
    if not isinstance(_m, list):
        # raise ValueError()
        return [[-1, -1]]

    # if a sublist of _m has more than 2 elements, it means that the model has not learned to predict the right format
    # substitute that sublist with [-1, -1]
    for i in range(len(_m)):
        if len(_m[i]) != 2:
            # print(f"Got a sublist with more or less than 2 elements!{_m[i]}")
            _m[i] = [-1, -1]

    return _m

def formatting(data_file, meta_file, out_file):
    data = load_json(data_file)
    meta_data = load_jsonl(meta_file)

    for d in data:
        # add a dummy confidence score
        d['pred_windows'] = [m + [1.0] for m in d['pred_windows']]

    new_meta_data = {}
    for meta_d in meta_data:
        new_meta_data[meta_d['qid']] = meta_d

    # print(len(meta_data), len(new_meta_data))
    # print(new_meta_data)

    def get_submission(data, meta_data):
        submissions = []
        for d in data:
            out = {}
            qid = int(d['qid'].split('_')[1])
            out["qid"] = qid
            out["query"] = meta_data[qid]["query"]
            out["vid"] = meta_data[qid]["vid"]
            for w in d["pred_windows"]:
                w[0] = float(w[0])
                w[1] = float(w[1])
            out["pred_relevant_windows"] = d["pred_windows"]

            out["pred_saliency_scores"] = [1.0] * len(d["pred_windows"])
            submissions.append(out)
        return submissions

    submission = get_submission(data, new_meta_data)
    save_jsonl(submission, out_file)

val_data_file = '../lavis/results/charades/CALCE_QVH_150_stage2-1/result/val_epochbest.json'
val_meta_file = '/data2/xiepeiyu/qvhighlights/annotations/highlight_val_release.jsonl'
val_out_file = '../submit/hl_val_submission.jsonl'

test_data_file = '../lavis/results/charades/CALCE_QVH_150_stage2-1/result/val_epochbest.json'
test_meta_file = '/data2/xiepeiyu/qvhighlights/annotations/highlight_val_release.jsonl'
test_out_file = '../submit/hl_val_submission.jsonl'

formatting(val_data_file,val_meta_file, val_out_file)
formatting(test_data_file,test_meta_file, test_out_file)



