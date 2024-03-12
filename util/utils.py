import numpy as np
import time


def get_entities(seq, suffix=False):
    """Gets entities from sequence.
    Args:
        seq (list): sequence of labels.
    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).
    """
    class2span = {}

    # for nested list
    if any(isinstance(s, list) for s in seq):
        seq = [item for sublist in seq for item in sublist + ['O']]

    prev_tag = 'O'
    prev_type = ''
    begin_offset = 0
    chunks = []
    for i, chunk in enumerate(seq + ['O']):
        if suffix:
            tag = chunk[-1]
            type_ = chunk.split('-')[0]
        else:
            tag = chunk[0]
            type_ = chunk.split('-', 1)[-1]

        if end_of_chunk(prev_tag, tag, prev_type, type_):
            chunks.append((prev_type, begin_offset, i-1))
        if start_of_chunk(prev_tag, tag, prev_type, type_):
            begin_offset = i
        prev_tag = tag
        prev_type = type_
    for chunk in chunks:
        cls, start, end = chunk
        if cls not in class2span:
            class2span[cls] = []
        class2span[cls].append([start, end])
    return class2span


def end_of_chunk(prev_tag, tag, prev_type, type_):
    """Checks if a chunk ended between the previous and current word.
    Args:
        prev_tag: previous chunk tag.
        tag: current chunk tag.
        prev_type: previous type.
        type_: current type.
    Returns:
        chunk_end: boolean.
    """
    chunk_end = False

    if prev_tag == 'E': chunk_end = True
    if prev_tag == 'S': chunk_end = True

    if prev_tag == 'B' and tag == 'B': chunk_end = True
    if prev_tag == 'B' and tag == 'S': chunk_end = True
    if prev_tag == 'B' and tag == 'O': chunk_end = True
    if prev_tag == 'I' and tag == 'B': chunk_end = True
    if prev_tag == 'I' and tag == 'S': chunk_end = True
    if prev_tag == 'I' and tag == 'O': chunk_end = True

    if prev_tag != 'O' and prev_tag != '.' and prev_type != type_:
        chunk_end = True

    return chunk_end


def start_of_chunk(prev_tag, tag, prev_type, type_):
    """Checks if a chunk started between the previous and current word.
    Args:
        prev_tag: previous chunk tag.
        tag: current chunk tag.
        prev_type: previous type.
        type_: current type.
    Returns:
        chunk_start: boolean.
    """
    chunk_start = False

    if tag == 'B': chunk_start = True
    if tag == 'S': chunk_start = True

    if prev_tag == 'E' and tag == 'E': chunk_start = True
    if prev_tag == 'E' and tag == 'I': chunk_start = True
    if prev_tag == 'S' and tag == 'E': chunk_start = True
    if prev_tag == 'S' and tag == 'I': chunk_start = True
    if prev_tag == 'O' and tag == 'E': chunk_start = True
    if prev_tag == 'O' and tag == 'I': chunk_start = True

    if tag != 'O' and tag != '.' and prev_type != type_:
        chunk_start = True

    return chunk_start


def now():
    return str(time.strftime('%Y-%m-%d %H:%M:%S'))


def remove_duplicate_beams(beams):
    all_path = set()
    res_beams = []
    for beam in beams:
        path = []
        sequence = beam.to_sequence()
        for node in sequence:
            start, end = node.start, node.end
            path.append((start, end))
        # import ipdb; ipdb.set_trace()
        path = sorted(path, key=lambda x:x[0]+100*x[1])
        path.append(beam.get_sequence_score())
        path = tuple(path)
        if path not in all_path:
            res_beams.append(beam)
            all_path.add(path)
    # print(beams)
    # print(res_beams)
    return res_beams


def warmup_linear(global_step, warmup_step):
    if global_step < warmup_step:
        return global_step / warmup_step
    else:
        return 1.0


def f_score_decay(si, sj, score, u, k):
    si = set([i for i in range(si[0], si[1]+1)])
    sj = set([j for j in range(sj[0], sj[1]+1)])
    IOU = len(si & sj) / len(si | sj)
    if IOU >= k:
        return score * u
    else:
        return score


class BeamNode():
    def __init__(self, parent, tuple):
        super(BeamNode, self).__init__()
        self.start = tuple[0]
        self.end = tuple[1]
        self.score = tuple[2]
        self.pred = tuple[3]
        self.parent = parent
        self._sequence = None

    def get_sorted_valid_tuples_soft_one(self, all_tuples, k, u, delta):
        sequence = self.to_sequence()
        valid_tuples = []
        for tuple_ in all_tuples:
            span_l, span_r, span_score, span_pred = tuple_
            is_in_beam = False
            for node in sequence:
                l, r = node.start, node.end
                if span_l == l and span_r == r:
                    is_in_beam = True
                    break
                span_score = f_score_decay((span_l, span_r), (l, r), span_score, u, k)
            
            # import ipdb; ipdb.set_trace()
            if not is_in_beam and span_score >= delta:
                valid_tuples.append((span_l, span_r, span_score, span_pred))

        valid_tuples = sorted(valid_tuples, key=lambda x: x[-2], reverse=True)
        return valid_tuples        

    def to_sequence(self):
        # Return sequence of nodes from root to current node.
        if not self._sequence:
            self._sequence = []
            current_node = self
            while current_node:
                self._sequence.insert(0, current_node)
                current_node = current_node.parent
        return self._sequence

    def get_sequence_score(self):
        return np.mean(np.array([s.score for s in self.to_sequence()]))

    def get_sequence_total_score(self):
        return np.sum(np.array([s.score for s in self.to_sequence()]))

    def get_tuples(self):
        return [(s.start, s.end, s.pred) for s in self.to_sequence()]


def beam_search_soft_nms(entity_tuples, beam_size, k, u, delta):
    out_entitys = []
    for each_entity_tuples in entity_tuples:
        each_entity_tuples = sorted(each_entity_tuples, key=lambda x: x[-2], reverse=True)
        if len(each_entity_tuples) == 0:
            out_entitys.append([])
            continue
        pre_beams =[BeamNode(None, each_entity_tuples[0])]
        all_exist_beam = set()
        while True:
            now_beams = []
            is_update = False
            for beam in pre_beams:
                # import ipdb; ipdb.set_trace()
                valid_tuples = beam.get_sorted_valid_tuples_soft_one(each_entity_tuples, k, u, delta)
                # can not expand, keep it                                                           
                if len(valid_tuples) == 0:
                    now_beams.append(beam)
                for valid_tuple in valid_tuples:
                    new_node = BeamNode(beam, valid_tuple)
                    if new_node not in all_exist_beam:
                        now_beams.append(new_node)
                        all_exist_beam.add(new_node)
            # print(len(now_beams))
            now_beams = remove_duplicate_beams(now_beams)
            now_beams = sorted(now_beams, key=lambda n: n.get_sequence_score(), reverse=True)[:beam_size]
            for beam in now_beams:
                if beam not in pre_beams:
                    is_update = True
            if not is_update:
                break
            pre_beams = now_beams
        result = now_beams[0].get_tuples()
        out_entitys.append(result)
    return out_entitys


def get_p_r_f1(correct_cnt, pred_cnt, label_cnt):
    precision = correct_cnt / (pred_cnt + 1e-12)
    recall = correct_cnt / (label_cnt + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)

    return precision, recall, f1


def get_fp(false_cnt, all_fp_span_cnt, all_fp_type_cnt):
    fp_span = all_fp_span_cnt / (false_cnt + 1e-6)
    fp_type = all_fp_type_cnt / (false_cnt + 1e-6)
    return fp_span, fp_type


def metrics_by_entity_tuples(gold_tuples, pred_tuples):
    correct_cnt = label_cnt = pred_cnt = 0
    fp_span_cnt = fp_type_cnt = 0
    pred_span_cnt = correct_span_cnt = 0
    false_cnt = 0
    for gold_tuple, pred_tuple in zip(gold_tuples, pred_tuples):
        pred_cnt += len(pred_tuple)
        label_cnt += len(gold_tuple)
        correct_cnt += len(pred_tuple & gold_tuple)
        pred_span_cnt += len(pred_tuple)
        if 1 > 2:
            print("***********")
            print("Final result: ")
            print(gold_tuple)
            print(pred_tuple)
            print("***********")
        for pred in pred_tuple:
            fp_span_flag = True
            fp_type_flag = True
            for gold in gold_tuple:
                if pred[0:2] == gold[0:2] and pred[2] == gold[2]:  # 都对
                    fp_span_flag = False
                    fp_type_flag = False
                    correct_span_cnt += 1
                    break
                elif pred[0:2] == gold[0:2] and pred[2] != gold[2]:  # span对 type错
                    fp_span_flag = False
                    correct_span_cnt += 1
                    break
            if fp_span_flag:  # span分错了
                fp_span_cnt += 1
                false_cnt += 1
            if fp_span_flag is not True and fp_type_flag:  # span对了但是type错了
                fp_type_cnt += 1
                false_cnt += 1

    return correct_cnt, pred_cnt, label_cnt, false_cnt, fp_span_cnt, fp_type_cnt, correct_span_cnt, pred_span_cnt


def overlap_score(b1, e1, b2, e2):
    if b1 > e2 or e1 < b2:
        return 0
    return (min(e1, e2) - max(b1, b2) + 1) / (e2 - b2 + 1)


def metrics_by_span_detector_V0(gold_tuples, span_list):
    pred_span_cnt = correct_span_cnt = 0
    label_cnt = 0
    long_cnt = 0
    short_cnt = 0
    wo_overlap_cnt = 0
    share_begin_cnt = 0
    share_end_cnt = 0
    for gold_tuple, pred_span in zip(gold_tuples, span_list):
        label_cnt += len(gold_tuple)
        pred_span_cnt += len(pred_span)

        if len(pred_span) > 0:
            for span in pred_span:
                long = short = wo_overlap = share_begin = share_end = 0
                max_overlap_score = -1.0
                for gold in gold_tuple:
                    if span[0] == gold[0] and span[1] == gold[1]:
                        correct_span_cnt += 1
                        long = short = wo_overlap = share_begin = share_end = 0
                        break
                    cur_overlap_score = overlap_score(span[0], span[1], gold[0], gold[1])
                    if max_overlap_score < cur_overlap_score:
                        max_overlap_score = cur_overlap_score
                        long = short = wo_overlap = 0
                        if span[0] == gold[0]:
                            share_begin += 1
                        if span[1] == gold[1]:
                            share_end += 1
                        if cur_overlap_score == 0.0:
                            wo_overlap += 1
                        elif (span[1] - span[0] + 1) > (gold[1] - gold[0] + 1):
                            long += 1
                        else:
                            short += 1
                long_cnt += long
                short_cnt += short
                wo_overlap_cnt += wo_overlap
                share_begin_cnt += share_begin
                share_end_cnt += share_end

    return label_cnt, correct_span_cnt, pred_span_cnt, long_cnt, short_cnt, wo_overlap_cnt, share_begin_cnt, share_end_cnt

def metrics_by_span_detector(gold_tuples, span_list):
    pred_span_cnt = correct_span_cnt = 0
    label_cnt = 0
    long_cnt = 0
    short_cnt = 0
    wo_overlap_cnt = 0
    share_begin_cnt = 0
    share_end_cnt = 0
    for gold_tuple, pred_span in zip(gold_tuples, span_list):
        label_cnt += len(gold_tuple)
        pred_span_cnt += len(pred_span)
        if len(pred_span) > 0:
            for span in pred_span:
                for gold in gold_tuple:
                    if span[0] == gold[0] and span[1] == gold[1]:
                        correct_span_cnt += 1
                        break

    return label_cnt, correct_span_cnt, pred_span_cnt, long_cnt, short_cnt, wo_overlap_cnt, share_begin_cnt, share_end_cnt

def metrics_by_entity_tuples_v0(gold_tuples, pred_tuples):
    correct_cnt = label_cnt = pred_cnt = 0
    for gold_tuple, pred_tuple in zip(gold_tuples, pred_tuples):

        pred_cnt += len(pred_tuple)
        label_cnt += len(gold_tuple)
        correct_cnt += len(pred_tuple & gold_tuple)

    return correct_cnt, pred_cnt, label_cnt