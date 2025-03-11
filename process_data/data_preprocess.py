import os
import json


def save_json(content, save_path):
    # if no such directory, create one
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    with open(save_path, 'w') as f:
        f.write(json.dumps(content))
def load_jsonl(filename):
    with open(filename, "r") as f:
        return [json.loads(l.strip("\n")) for l in f.readlines()]
def load_json(filename):
    with open(filename, "r") as f:
        return json.load(f)



save_float = False
relative_time = False

# process QVHighlights
train_path = '/data2/xiepeiyu/qvhighlights/annotations/highlight_train_release.jsonl'
val_path = '/data2/xiepeiyu/qvhighlights/annotations/highlight_val_release.jsonl'
test_path = '/data2/xiepeiyu/qvhighlights/annotations/highlight_test_release.jsonl'
train_out_path = '../lavis/datasets/annotations/qvh/train.json'
val_out_path = '../lavis/datasets/annotations/qvh/val.json'
test_out_path = '../lavis/datasets/annotations/qvh/test_dummy.json'

train_item_path = '../lavis/datasets/annotations/qvh/items/train.json'
val_item_path = '../lavis/datasets/annotations/qvh/items/val.json'
test_item_path = '../lavis/datasets/annotations/qvh/items/test.json'

def process_QVH(in_path, out_path, item_path, relative_time=False, save_float=False, is_test=False):
    data = load_jsonl(in_path)
    item_anno = load_json(item_path)
    new_data = []
    for d in data:
        sample = {}
        sample['video'] = d['vid']
        sample['qid'] = 'QVHighlight_' + str(d['qid'])
        sample['query'] = d['query']
        duration = d['duration']
        sample['duration'] = duration
        sample['items'] = item_anno[sample['qid']]['items']

        if not is_test:
            windows = d['relevant_windows']
            if relative_time:
                relative_time_windows = []
                for window in windows:
                    start = window[0] / duration
                    end = window[1] / duration

                    if save_float:
                        relative_time_windows.append([round(start, 2), round(end, 2)])
                    else:
                        relative_time_windows.append([int(round(start, 2) * 100), int(round(end, 2) * 100)])
                sample['relevant_windows'] = relative_time_windows
            else:
                sample['relevant_windows'] = windows
        else:
            sample['relevant_windows'] = [[0, 150]] # dummy value

        new_data.append(sample)

    save_json(new_data, out_path)


process_QVH(train_path,train_out_path, train_item_path, relative_time=relative_time, save_float=save_float)
process_QVH(val_path,val_out_path, val_item_path,  relative_time=relative_time, save_float=save_float)
process_QVH(test_path,test_out_path, test_item_path, relative_time=relative_time, save_float=save_float, is_test=True)


# process Charades_STA
train_path = '/data2/xiepeiyu/Charades-STA/annotations/lavis/charades_sta_train_tvr_format.jsonl'
test_path  = '/data2/xiepeiyu/Charades-STA/annotations/lavis/charades_sta_test_tvr_format.jsonl'
train_out_path = '../lavis/datasets/annotations/charades/train.json'
test_out_path = '../lavis/datasets/annotations/charades/test.json'

train_item_path = '../lavis/datasets/annotations/charades/items/train.json'
test_item_path = '../lavis/datasets/annotations/charades/items/test.json'

def process_charades(in_path, out_path, item_path, save_float=False, is_test=False):
    data = load_jsonl(in_path)
    item_anno = load_json(item_path)
    new_data = []
    for d in data:
        relevant_windows = d['relevant_windows']
        for i, relevant_window in enumerate(relevant_windows):
            if save_float or is_test:
                relevant_window = [float(relevant_window[0]), float(relevant_window[1])]
            else:
                relevant_window = [round(float(relevant_window[0])), round(float(relevant_window[1]))]
            relevant_windows[i] = relevant_window
        qid = 'Charades_'+ str(d['qid'])
        sample = {
            'qid': qid,
            'video': d['vid'],
            'duration': round(float(d['duration'])),
            'relevant_windows': relevant_windows,
            'query': d['query'],
            'items': item_anno[qid]['items']
        }
        new_data.append(sample)
    save_json(new_data,out_path)

process_charades(train_path,train_out_path, train_item_path, save_float=save_float)
process_charades(test_path,test_out_path, test_item_path, save_float=save_float, is_test=True)


