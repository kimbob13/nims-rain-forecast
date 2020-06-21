import os
import argparse
import numpy as np
import pandas as pd
        
def parse_args():
    parser = argparse.ArgumentParser(description='NIMS converter from confusion matrix to evaluation metric')
    
    parser.add_argument('--experiment_name', type=str, help='experiment name that you want to convert')
    parser.add_argument('--baseline_name', type=str, help='baseline experiment name')

    args = parser.parse_args()

    return args

def convert_micro_confusion_matrix(experiment_name, baseline_name):
    # From micro confusion matrix
    micro_pd = pd.read_csv(os.path.join('./results',
                                        'eval',
                                        'micro-{}.csv'.format(experiment_name)), index_col=0)

    H = micro_pd['pred_yes'][0] # Hits
    M = micro_pd['pred_no'][0] # Misses
    F = micro_pd['pred_yes'][1] # False alarms
    C = micro_pd['pred_no'][1] # Correct negatives

    # 1. Probability of Detection, POD
    pod = H / (H+M)

    # 2. False Alarm Ratio, FAR
    far = F / (H+F)

    # 3. Post Agreement, PAG
    pag = H / (H+F)

    # 4. Bias
    bias = (H+F) / (H+M)

    # 5. Accuracy
    acc = (H+C) / (H+M+F+C)

    # 6. Critical Success Index (or Threat Score), CSI (or TS)
    csi = H / (H+M+F)

    # 7. Heidke Skill Score, HSS
    correct_std = ((H+M)*(H+F) + (C+M)*(C+F)) / (H+M+F+C)
    hss = (H+C-correct_std) / (H+M+F+C-correct_std)

    # 8. Kuipers Skill Score, KSS
    kss = (H*C-M*F) / ((H+M)*(F+C))

    if baseline_name is None:
        return pod, far, pag, bias, acc, csi, hss, kss
    else:
        # 9. Improvement Against Standard
        baseline_micro_pd = pd.read_csv(os.path.join('./results',
                                                     'eval',
                                                     'micro-{}.csv'.format(baseline_name)))

        baseline_H = baseline_micro_pd['pred_yes'][0] # Hits
        baseline_M = baseline_micro_pd['pred_no'][0] # Misses
        baseline_F = baseline_micro_pd['pred_yes'][1] # False alarms
        baseline_C = baseline_micro_pd['pred_no'][1] # Correct negatives

        baseline_csi = baseline_H / (baseline_H+baseline_M+baseline_F)
        improvement = (csi-baseline_csi) / baseline_csi
        return pod, far, pag, bias, acc, csi, hss, kss, improvement
        
def convert_macro_confusion_matrix(experiment_name, baseline_name):
    # From macro confusion matrix
    macro_pd = pd.read_csv(os.path.join('./results',
                                        'eval',
                                        'macro-{}.csv'.format(experiment_name)), index_col=0)

    """
    # 1. Accuracy
    total_value = np.sum(np.array(macro_pd)[:4, :4])
    digonal_value = np.sum(np.array(macro_pd)[range(4), range(4)])
    acc = digonal_value / total_value

    # 2. Heidke Skill Score, HSS
    correct_std = ((H+M)*(H+F) + (C+M)*(C+F)) / (H+M+F+C)
    hss = (H+C-correct_std) / (H+M+F+C-correct_std)

    # 3. Improvement Against Standard
    if baseline_name is None:
        pass
    else:
        baseline_micro_pd = pd.read_csv(os.path.join('./results',
                                                     'eval',
                                                     'micro-{}.csv'.format(baseline_name)))

        baseline_H = baseline_micro_pd['pred_yes'][0] # Hits
        baseline_M = baseline_micro_pd['pred_no'][0] # Misses
        baseline_F = baseline_micro_pd['pred_yes'][1] # False alarms
        baseline_C = baseline_micro_pd['pred_no'][1] # Correct negatives

        baseline_csi = baseline_H / (baseline_H+baseline_M+baseline_F)
        improvement = (csi-baseline_csi) / baseline_csi
    """
    


if __name__ == '__main__':
    args = parse_args()
    
    experiment_name = args.experiment_name
    baseline_name = args.baseline_name
    
    evaluation_metric_filename = os.path.join('./results',
                                              'eval',
                                              'eval_metric-{}.txt'.format(experiment_name))
    
    micro_metric_name = ['Probability of Detection',
                         'False Alarm Ratio',
                         'Post Agreement',
                         'Bias',
                         'Accuracy',
                         'Critical Success index',
                         'Heidke Skill Score',
                         'Kuipers Skill Score']
    
    # macro_metric_name = []
    
    if baseline_name is not None:
        micro_metric_name.append('Improvement Against Standard')
        # macro_metric_name.append('Improvement Against Standard')
    
    with open(evaluation_metric_filename, "w") as f:
        f.write("Evaluation metric from micro confusion matrix\n")
        f.write("=============================================\n")
        micro_metric = convert_micro_confusion_matrix(experiment_name, baseline_name)
        for name, metric in list(zip(micro_metric_name, micro_metric)):
            f.write("{}: {}\n".format(name, metric))
        """
        f.write("=============================================\n\n")
        f.write("Evaluation metric from macro confusion matrix\n")
        f.write("=============================================\n")
        macro_metric = convert_macro_confusion_matrix(experiment_name, baseline_name)
        for name, metric in list(zip(macro_metric_name, macro_metric)):
            f.write("{}: {}\n".format(name, metric))
        f.write("=============================================")
        """