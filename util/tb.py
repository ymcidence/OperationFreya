from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensorflow.python.framework import tensor_util
from tensorflow.python.summary.summary_iterator import summary_iterator

def get_keys(path):
    ea = EventAccumulator(path)
    ea.Reload()
    return ea.Tags()['tensors']
def sum_log(path, blocking=['adj']):
    
    tags = get_keys(path)
    vals = dict()
    steps = dict()
    for t in tags:
        valid_tag = True
        for b in blocking:
            if str(t).find(b) >= 0:
                valid_tag = False
                break
        if valid_tag:
            vals[t] = []
    
    try:
        for e in summary_iterator(path):
            for v in e.summary.value:
                if vals.get(v.tag, None) is not None:
                    vals[v.tag].append(tensor_util.MakeNdarray(v.tensor))
    
    # Dirty catch of DataLossError
    except:
        print('Event file possibly corrupt: {}'.format(path))
    
    return vals