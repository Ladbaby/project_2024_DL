import os
import numpy as np
import distance
from nltk.translate.bleu_score import sentence_bleu
from dateutil import rrule
from datetime import datetime
import time


pred_dir = './mathmatical-expressions-recognition/evaluation/submit_results'
target_file_path = os.path.join('./mathmatical-expressions-recognition/evaluation/submit_results','ground_truth.txt')

test_file_dir =  './images/datasets/test/labels'
test_ids_path =  './images/datasets/test_ids.txt'

summary_dir = './mathmatical-expressions-recognition/evaluation/evaluate_summary'

def main():
    # create target file
    if not os.path.isfile(target_file_path):
        with open(target_file_path,'w') as fout, open(test_ids_path,'r') as fin:
            for one_line in fin.readlines():
                one_ids = int(one_line)
                with open(os.path.join(test_file_dir,str(one_ids)+'.txt'),'r') as f:
                    one_label = str(f.readlines()[0]).strip()
                    fout.write(one_label+'\n')

    # evaluate each summited files
    summary_dict = {
        '1': # model 1
        {   
            # group_id: {}
        }, 
        '2': # model 2
        {
            # group_id: {}
        }  
    }
    for pred_file in os.listdir(pred_dir):
        if pred_file == 'ground_truth.txt':
            continue
        pred_file_path = os.path.join(pred_dir,pred_file)
        group_id, model_id = pred_file[:-4].split('-')[0], pred_file[:-4].split('-')[1]
        with open(pred_file_path,'r') as f_pred, open(target_file_path,'r') as f_gt:
            pred_lst = []
            gt_lst = []
            pred_lines = f_pred.readlines()
            gt_lines = f_gt.readlines()
            if len(pred_lines)!=len(gt_lines):
                print('Lengths are not equal! len(pred_lines)=%d; len(gt_lines)=%d'%(len(pred_lines),len(gt_lines)))
                # bleu4, Edit_Distance, Exact_Match = 0.0, 0.0, 0.0

            for pred_line, gt_line in zip(pred_lines, gt_lines):
                pred_line = pred_line.strip().replace(' ','').replace('\t','').replace('\r','').replace('\n','')
                gt_line = gt_line.strip().replace(' ','').replace('\t','').replace('\r','').replace('\n','')
                if gt_line == 'errormathpix':
                    continue
                pred_lst.append(pred_line)
                gt_lst.append(gt_line)
            bleu4, Edit_Distance, Exact_Match = evaluate(pred_lst, gt_lst)

            print('%s : bleu4=%.4f; Edit_Distance=%.4f; Exact_Match=%.4f'%(pred_file,bleu4,Edit_Distance,Exact_Match))
            summary_dict[model_id][group_id] = {'bleu_score':bleu4,'edit_distance_score':Edit_Distance,'exact_match':Exact_Match}

    # write summary results
    summary_file = os.path.join(summary_dir, '%s.txt'%(time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime())))
    with open(summary_file,'w') as f_summary:
        f_summary.write('='*120+'\n')
        f_summary.write('{:^200}\n'.format('Model 1 Leaderboard'))
        f_summary.write('='*120+'\n')
        f_summary.write('# {:<30}{:<30}{:<30}{:<30}{:<30}{:<30}\n'.format('Rank','Group_ID','OVERALL_SCORE','(1)BLEU_SCORE','(2)EDIT_DISTANCE_SCORE','(3)Exact_Match'))
        sorted_group_id = sorted([(k,v) for k,v in summary_dict['1'].items()], key=lambda x: (x[1]['bleu_score']+x[1]['edit_distance_score']+x[1]['exact_match'])/3, reverse=True)
        rank_id = 1
        group_cnt = len(sorted_group_id)
        for group_id, scores_dict in sorted_group_id:
            f_summary.write('# {:<35}{:<35}{:<39}{:<42}{:<39}{:<33}\n'.format(
                    '%2s/%2s'%(rank_id,group_cnt),
                    '%2s'%(group_id),
                    '%7.4f'%((scores_dict['bleu_score']+scores_dict['edit_distance_score']+scores_dict['exact_match'])/3),
                    '%7.4f'%(scores_dict['bleu_score']),
                    '%7.4f'%(scores_dict['edit_distance_score']),
                    '%7.4f'%(scores_dict['exact_match'])
                )
            )
            rank_id+=1
        f_summary.write('='*120+'\n')
        f_summary.write('{:^200}\n'.format('END'))
        f_summary.write('='*120+'\n\n\n')

        f_summary.write('='*120+'\n')
        f_summary.write('{:^200}\n'.format('Model 2 Leaderboard'))
        f_summary.write('='*120+'\n')
        f_summary.write('# {:<30}{:<30}{:<30}{:<30}{:<30}{:<30}\n'.format('Rank','Group_ID','OVERALL_SCORE','(1)BLEU_SCORE','(2)EDIT_DISTANCE_SCORE','(3)Exact_Match'))
        sorted_group_id = sorted([(k,v) for k,v in summary_dict['2'].items()], key=lambda x: (x[1]['bleu_score']+x[1]['edit_distance_score']+x[1]['exact_match'])/3, reverse=True)
        rank_id = 1
        group_cnt = len(sorted_group_id)
        for group_id, scores_dict in sorted_group_id:
            f_summary.write('# {:<35}{:<35}{:<39}{:<42}{:<39}{:<33}\n'.format(
                    '%2s/%2s'%(rank_id,group_cnt),
                    '%2s'%(group_id),
                    '%7.4f'%((scores_dict['bleu_score']+scores_dict['edit_distance_score']+scores_dict['exact_match'])/3),
                    '%7.4f'%(scores_dict['bleu_score']),
                    '%7.4f'%(scores_dict['edit_distance_score']),
                    '%7.4f'%(scores_dict['exact_match'])
                )
            )
            rank_id+=1
        f_summary.write('='*120+'\n')
        f_summary.write('{:^200}\n'.format('END'))
        f_summary.write('='*120+'\n\n\n')

def evaluate(references, hypotheses):
    #用于在验证集上计算各种评价指标指导模型早停
    # Calculate scores
    bleu4 = 0.0
    for i,j in zip(references,hypotheses):
        bleu4 += max(sentence_bleu([i],j),0.01)
    bleu4 = bleu4/len(references)
    bleu4 = bleu4*100
    Edit_Distance = edit_distance(references, hypotheses)
    Exact_Match = np.mean([1.0 if r==h else 0.0 for r,h in zip(references, hypotheses)])*100
    return bleu4, Edit_Distance, Exact_Match

def edit_distance(references, hypotheses):
    """Computes Levenshtein distance between two sequences.
    Args:
        references: list of list of token (one hypothesis)
        hypotheses: list of list of token (one hypothesis)
    Returns:
        1 - levenshtein distance: (higher is better, 1 is perfect)
    """
    d_leven, len_tot = 0, 0
    for ref, hypo in zip(references, hypotheses):
        d_leven += distance.levenshtein(ref, hypo)
        len_tot += float(max(len(ref), len(hypo)))

    return (1. - d_leven / len_tot)*100
    
# pred = ['A D^{2}=P C^{2}+A C^{2}-2 D C \cdot A c \cdot \cos 30^{\circ}']
# tgt = ['A D^{2}=P C^{2} + A C^{2}-2 D C \cdot A c \cdot \cos 30^{\circ}']
# print(evaluate(pred,tgt))

main()