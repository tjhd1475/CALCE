import json, argparse

parser = argparse.ArgumentParser()
parser.add_argument('--eval', action='store_true')



def merge(orig_file, stage1_file, out_file):
    with open(stage1_file, 'r') as f:
        stage1_result = json.load(f)

    stage1_result = {d['qid']: d for d in stage1_result}

    with open(orig_file, 'r') as f:
        data = json.load(f)

    for d in data:
        qid = d['qid']
        d['stage1_result'] = stage1_result[qid]['pred_windows']

    with open(out_file, 'w') as f:
        json.dump(data, f)


if __name__ == '__main__':
    args = parser.parse_args()
    eval_only = args.eval
    # QVHighlights

    train_orig_file = './lavis/datasets/annotations/qvh/train.json'
    train_out_file = './lavis/datasets/annotations/qvh/train_stage2.json'
    train_stage1_file = './results/qvh/CALCE_QVH_75_stage1-1/result/train_epochbest.json'

    val_orig_file = './lavis/datasets/annotations/qvh/val.json'
    val_out_file = './lavis/datasets/annotations/qvh/val_stage2.json'
    val_stage1_file = './lavis/results/qvh/CALCE_QVH_75_stage1-1/result/val_epochbest.json'

    test_orig_file = './lavis/datasets/annotations/qvh/test_dummy.json'
    test_out_file = './lavis/datasets/annotations/qvh/test_dummy_stage2.json'
    test_stage1_file = './lavis/results/qvh/CALCE_QVH_75_stage1-1/result/test_epochbest.json'

    if not eval_only:
        merge(train_orig_file, train_stage1_file, train_out_file)
    merge(val_orig_file, val_stage1_file, val_out_file)
    merge(test_orig_file, test_stage1_file, test_out_file)

    # Charades-STA

    train_orig_file = './lavis/datasets/annotations/charades/train.json'
    train_out_file = './lavis/datasets/annotations/charades/train_stage2.json'
    train_stage1_file = './lavis/results/charades/CALCE_Charades_60_stage1-1/result/train_epochbest.json'

    test_orig_file = './lavis/datasets/annotations/charades/test.json'
    test_out_file = './lavis/datasets/annotations/charades/test_stage2.json'
    test_stage1_file = './lavis/results/charades/CALCE_Charades_60_stage1-1/result/test_epochbest.json'

    if not eval_only:
        merge(train_orig_file, train_stage1_file, train_out_file)
    merge(test_orig_file, test_stage1_file, test_out_file)
