import os.path as osp
from utils import load_file

map_name = {'D': 'Description', 'E': 'Explaination', 'PA': 'Predictive-Answer', 'CA': 'Counterfactual-Answer', 'PR': 'Predictive-Reason', 'CR': 'Counterfactual-Reason', 'P':'Predictive', 'C': 'Counterfactual'}

def accuracy_metric(result_file, qtype):
    if qtype == -1:
        accuracy_metric_all(result_file)
    if qtype == 0:
        accuracy_metric_q0(result_file)
    if qtype == 1:
        accuracy_metric_q1(result_file)
    if qtype == 2:
        accuracy_metric_q2(result_file)
    if qtype == 3:
        accuracy_metric_q3(result_file)

def accuracy_metric_q0(result_file):
    preds = list(load_file(result_file).items())
    group_acc = {'D': 0}
    group_cnt = {'D': 0}
    all_acc = 0
    all_cnt = 0
    for idx in range(len(preds)):
        id_qtypes = preds[idx]
        answer = id_qtypes[1]['answer']
        pred = id_qtypes[1]['prediction']
        group_cnt['D'] += 1
        all_cnt += 1
        if answer == pred:
            group_acc['D'] += 1
            all_acc += 1
    for qtype, acc in group_acc.items(): #
        print('{0:21} ==> {1:6.2f}%'.format(map_name[qtype], acc*100.0/group_cnt[qtype]))
    print('{0:21} ==> {1:6.2f}%'.format('Acc', all_acc*100.0/all_cnt))

def accuracy_metric_q1(result_file):
    preds = list(load_file(result_file).items())
    group_acc = {'E': 0}
    group_cnt = {'E': 0}
    all_acc = 0
    all_cnt = 0
    for idx in range(len(preds)):
        id_qtypes = preds[idx]
        answer = id_qtypes[1]['answer']
        pred = id_qtypes[1]['prediction']
        group_cnt['E'] += 1
        all_cnt += 1
        if answer == pred:
            group_acc['E'] += 1
            all_acc += 1
    for qtype, acc in group_acc.items(): #
        print('{0:21} ==> {1:6.2f}%'.format(map_name[qtype], acc*100.0/group_cnt[qtype]))
    print('{0:21} ==> {1:6.2f}%'.format('Acc', all_acc*100.0/all_cnt))

def accuracy_metric_q2(result_file):
    preds = list(load_file(result_file).items())
    qtype2short = ['PA', 'PR', 'P']
    group_acc = {'PA': 0, 'PR': 0, 'P': 0}
    group_cnt = {'PA': 0, 'PR': 0, 'P': 0}
    all_acc = 0
    all_cnt = 0
    for idx in range(len(preds)//2):
        id_qtypes = preds[idx*2:(idx+1)*2]
        qtypes = [0, 1]
        answer = [ans_pre[1]['answer'] for ans_pre in id_qtypes]
        pred = [ans_pre[1]['prediction'] for ans_pre in id_qtypes]
        for i in range(2):
            group_cnt[qtype2short[qtypes[i]]] += 1
            if answer[i] == pred[i]:
                group_acc[qtype2short[qtypes[i]]] += 1
        group_cnt['P'] += 1
        all_cnt += 1
        if answer[0] == pred[0] and answer[1] == pred[1]:
            group_acc['P'] += 1
            all_acc += 1
    for qtype, acc in group_acc.items(): #
        print('{0:21} ==> {1:6.2f}%'.format(map_name[qtype], acc*100.0/group_cnt[qtype]))
    print('{0:21} ==> {1:6.2f}%'.format('Acc', all_acc*100.0/all_cnt))

def accuracy_metric_q3(result_file):
    preds = list(load_file(result_file).items())
    qtype2short = ['CA', 'CR', 'C']
    group_acc = {'CA': 0, 'CR': 0, 'C': 0}
    group_cnt = {'CA': 0, 'CR': 0, 'C': 0}
    all_acc = 0
    all_cnt = 0
    for idx in range(len(preds)//2):
        id_qtypes = preds[idx*2:(idx+1)*2]
        qtypes = [0, 1]
        answer = [ans_pre[1]['answer'] for ans_pre in id_qtypes]
        pred = [ans_pre[1]['prediction'] for ans_pre in id_qtypes]
        for i in range(2):
            group_cnt[qtype2short[qtypes[i]]] += 1
            if answer[i] == pred[i]:
                group_acc[qtype2short[qtypes[i]]] += 1
        group_cnt['C'] += 1
        all_cnt += 1
        if answer[0] == pred[0] and answer[1] == pred[1]:
            group_acc['C'] += 1
            all_acc += 1
    for qtype, acc in group_acc.items(): #
        print('{0:21} ==> {1:6.2f}%'.format(map_name[qtype], acc*100.0/group_cnt[qtype]))
    print('{0:21} ==> {1:6.2f}%'.format('Acc', all_acc*100.0/all_cnt))

def accuracy_metric_all(result_file):
    preds = list(load_file(result_file).items())
    qtype2short = ['D', 'E', 'PA', 'PR', 'CA', 'CR', 'P', 'C']
    group_acc = {'D': 0, 'E': 0, 'PA': 0, 'PR': 0, 'CA': 0, 'CR': 0, 'P': 0, 'C': 0}
    group_cnt = {'D': 0, 'E': 0, 'PA': 0, 'PR': 0, 'CA': 0, 'CR': 0, 'P': 0, 'C': 0}
    all_acc = 0
    all_cnt = 0
    for idx in range(len(preds)//6):
        id_qtypes = preds[idx*6:(idx+1)*6]
        qtypes = [int(id_qtype[0].split('_')[-1]) for id_qtype in id_qtypes]
        answer = [ans_pre[1]['answer'] for ans_pre in id_qtypes]
        pred = [ans_pre[1]['prediction'] for ans_pre in id_qtypes]
        for i in range(6):
            group_cnt[qtype2short[qtypes[i]]] += 1
            if answer[i] == pred[i]:
                group_acc[qtype2short[qtypes[i]]] += 1
        group_cnt['C'] += 1
        group_cnt['P'] += 1
        all_cnt += 4
        if answer[0] == pred[0]:
            all_acc += 1
        if answer[1] == pred[1]:
            all_acc += 1
        if answer[2] == pred[2] and answer[3] == pred[3]:
            group_acc['P'] += 1
            all_acc += 1
        if answer[4] == pred[4] and answer[5] == pred[5]:
            group_acc['C'] += 1
            all_acc += 1
    for qtype, acc in group_acc.items(): #
        print('{0:21} ==> {1:6.2f}%'.format(map_name[qtype], acc*100.0/group_cnt[qtype]))
    print('{0:21} ==> {1:6.2f}%'.format('Acc', all_acc*100.0/all_cnt))

def main(result_file, qtype=-1):
    print('Evaluating {}'.format(result_file))

    accuracy_metric(result_file, qtype)


if __name__ == "__main__":
    result_file = 'path to results json'
    main(result_file, -1)
