from mmaction.apis import init_recognizer
import torch
import argparse
from tqdm import tqdm
import os
import numpy as np
import torch.nn as nn
import random
from sklearn import metrics
import torch.nn.functional as F
from scipy.stats import entropy
from dataloader_video_flow import EPICDOMAIN
# from dataloader_video_flow_hac import HACDOMAIN

to_np = lambda x: x.data.cpu().numpy()

def calc_aurc_eaurc(softmax_max, correct):
    # softmax = np.array(softmax)
    correctness = np.array(correct)
    # softmax_max = np.max(softmax, 1)

    sort_values = sorted(zip(softmax_max[:], correctness[:]), key=lambda x: x[0], reverse=True)
    sort_softmax_max, sort_correctness = zip(*sort_values)
    risk_li, coverage_li = coverage_risk(sort_softmax_max, sort_correctness)
    aurc, eaurc = aurc_eaurc(risk_li)

    return aurc, eaurc

def calc_fpr_aupr(softmax_max, correct):
    # softmax = np.array(softmax)
    correctness = np.array(correct)
    # softmax_max = np.max(softmax, 1)

    fpr, tpr, thresholds = metrics.roc_curve(correctness, softmax_max)
    auroc = metrics.auc(fpr, tpr)
    idx_tpr_95 = np.argmin(np.abs(tpr - 0.95))
    fpr_in_tpr_95 = fpr[idx_tpr_95]
    tnr_in_tpr_95 = 1 - fpr[np.argmax(tpr >= .95)]

    precision, recall, thresholds = metrics.precision_recall_curve(correctness, softmax_max)
    aupr_success = metrics.auc(recall, precision)
    aupr_err = metrics.average_precision_score(-1 * correctness + 1, -1 * softmax_max)

    return auroc, aupr_success, aupr_err, fpr_in_tpr_95, tnr_in_tpr_95

def coverage_risk(confidence, correctness):
    risk_list = []
    coverage_list = []
    risk = 0
    for i in range(len(confidence)):
        coverage = (i + 1) / len(confidence)
        coverage_list.append(coverage)

        if correctness[i] == 0:
            risk += 1

        risk_list.append(risk / (i + 1))

    return risk_list, coverage_list

# Calc aurc, eaurc
def aurc_eaurc(risk_list):
    r = risk_list[-1]
    risk_coverage_curve_area = 0
    optimal_risk_area = r + (1 - r) * np.log(1 - r)
    for risk_value in risk_list:
        risk_coverage_curve_area += risk_value * (1 / len(risk_list))

    aurc = risk_coverage_curve_area
    eaurc = risk_coverage_curve_area - optimal_risk_area
    return aurc, eaurc

def compute_doctor_scores(probs):
    """
    Compute DOCTOR scores from classifier logits.
    
    Given logits (batch_size x num_classes):
      - Compute softmax probabilities.
      - Compute b_g = sum(softmax^2) along classes.
      - Define r_alpha = (1 - b_g) / (b_g + 1e-8).
      - Compute Pe_b = 1 - max(softmax) along classes.
      - Define r_beta = Pe_b / (1 - Pe_b + 1e-8).
    
    Returns:
        r_alpha, r_beta: Tensors of shape (batch_size,)
    """
    # probs = F.softmax(logits, dim=1)
    b_g = torch.sum(probs ** 2, dim=1)
    r_alpha = (1 - b_g) / (b_g + 1e-8)
    
    return -r_alpha

def validate_one_step(model, clip, labels, flow, model_flow):
    clip = clip['imgs'].cuda().squeeze(1)
    labels = labels.cuda()
    flow = flow['imgs'].cuda().squeeze(1)

    with torch.no_grad():
        x_slow, x_fast = model.module.backbone.get_feature(clip)  # 16,1024,8,14,14
        v_feat = (x_slow.detach(), x_fast.detach())  # slow 16,1280,16,14,14, fast 16,128,64,14,14

        v_feat = model.module.backbone.get_predict(v_feat)
        v_predict, v_emd = model.module.cls_head(v_feat)

        f_feat = model_flow.module.backbone.get_feature(flow)  # 16,1024,8,14,14
        f_feat = model_flow.module.backbone.get_predict(f_feat)
        f_predict, f_emd = model_flow.module.cls_head(f_feat)

        predict = mlp_cls(v_emd, f_emd)
        feature = torch.cat((v_emd, f_emd), dim=1)

        if args.use_mfs:
            predict = predict[:, :-1]

    return predict, feature, v_predict, f_predict

class Encoder(nn.Module):
    def __init__(self, input_dim=2816, out_dim=8):
        super(Encoder, self).__init__()
        self.enc_net = nn.Linear(input_dim, out_dim)
  
    def forward(self, vfeat, afeat):
        feat = torch.cat((vfeat, afeat), dim=1)
        return self.enc_net(feat)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--datapath', type=str, default='/path/to/video_datasets/',
                        help='datapath')
    parser.add_argument('--bsz', type=int, default=16,
                        help='batch_size')
    parser.add_argument("--resumef", type=str, default='checkpoint.pt')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--appen", type=str, default='')
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--dataset", type=str, default='HMDB') # HMDB UCF Kinetics
    parser.add_argument('--T', type=int, default=1, help='temperature parameter')
    parser.add_argument('--score', default='msp', type=str, choices=['msp', 'energy', 'max-logit', 'entropy', 'var', 'maha', 'doctor'], help='score options')
    parser.add_argument('--use_mfs', action='store_true')
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    v_dim = 2304
    f_dim = 2048

    # init_distributed_mode(args)
    config_file = 'configs/recognition/slowfast/slowfast_r101_8x8x1_256e_kinetics400_rgb.py'
    config_file_flow = 'configs/recognition/slowonly/slowonly_r50_8x8x1_256e_kinetics400_flow.py'
        
    device = 'cuda:0' # or 'cpu'
    device = torch.device(device)

    if args.dataset == 'HMDB':
        num_class = 43
    elif args.dataset == 'Kinetics':
        num_class = 229
    elif args.dataset == 'Kinetics100':
        num_class = 100
    elif args.dataset == 'HAC':
        num_class = 7

    if args.use_mfs:
        num_class = num_class + 1

    # build the model from a config file and a checkpoint file
    model = init_recognizer(config_file, device=device, use_frames=True)
    model.cls_head.fc_cls = nn.Linear(v_dim, num_class).cuda()
    cfg = model.cfg
    model = torch.nn.DataParallel(model)

    model_flow = init_recognizer(config_file_flow, device=device,use_frames=True)
    model_flow.cls_head.fc_cls = nn.Linear(f_dim, num_class).cuda()
    cfg_flow = model_flow.cfg
    model_flow = torch.nn.DataParallel(model_flow)

    mlp_cls = Encoder(input_dim=v_dim+f_dim, out_dim=num_class)
    mlp_cls = mlp_cls.cuda()

    resume_file = args.resumef
    print("Resuming from ", resume_file)
    checkpoint = torch.load(resume_file)

    model.load_state_dict(checkpoint['model_state_dict'])
    model_flow.load_state_dict(checkpoint['model_flow_state_dict'])
    mlp_cls.load_state_dict(checkpoint['mlp_cls_state_dict'])

    model.eval()
    model_flow.eval()
    mlp_cls.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_dataset = EPICDOMAIN(split='test', cfg=cfg, cfg_flow=cfg_flow, datapath=args.datapath, dataset=args.dataset)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.bsz, num_workers=args.num_workers, shuffle=False,
                                                pin_memory=(device.type == "cuda"), drop_last=False)

    dataloaders = {'test': test_dataloader}
    splits = ['test']

    for split in splits:
        print(split)
        list_softmax = []
        list_correct = []
        acc = 0
        count = 0

        list_softmax_mfs = []

        all_v_conf, all_v_label = [], []
        all_f_conf, all_f_label = [], []
        all_combined_conf, all_combined_label = [], []
        all_gt_label = []

        for clip, spectrogram, labels, index in tqdm(dataloaders[split]):
            output, feature, output_v, output_f = validate_one_step(model, clip, labels, spectrogram, model_flow)
            
            _, predict = torch.max(output.detach().cpu(), dim=1)
            acc1 = (predict == labels).sum().item()
            acc += int(acc1)

            count += output.size()[0]

            if args.score == 'doctor':
                smax = F.softmax(output/args.T, dim=1)
                list_softmax.extend(to_np(compute_doctor_scores(smax)))
            else:
                if args.score == 'max-logit':
                    smax = to_np(output)
                else:
                    smax = to_np(F.softmax(output/args.T, dim=1))
                if args.score == 'energy':
                    list_softmax.extend(to_np((args.T*torch.logsumexp(output / args.T, dim=1))))  
                elif args.score == 'entropy':  
                    list_softmax.extend(-entropy(smax, axis = 1)) 
                elif args.score == 'var':
                    list_softmax.extend(np.var(smax, axis = 1))
                elif args.score in ['msp', 'max-logit']:
                    list_softmax.extend(np.max(smax, axis=1)) 

            pred = output.data.max(1, keepdim=True)[1]
            for j in range(len(pred)):
                if pred[j] == labels[j]:
                    cor = 1
                else:
                    cor = 0
                list_correct.append(cor)

        list_softmax = np.array(list_softmax)

        list_softmax_mfs = np.array(list_softmax_mfs)

        aurc, eaurc = calc_aurc_eaurc(list_softmax, list_correct)
        # fpr, aupr
        auroc, aupr_success, aupr, fpr, tnr = calc_fpr_aupr(list_softmax, list_correct)

        Acc = acc / float(count)

        print("score: ", args.score)
        print("AURC {0:.2f}".format(aurc * 1000))
        print("AUROC {0:.2f}".format(auroc * 100))
        print('FPR95 {0:.2f}'.format(fpr * 100))
        print('Acc {0:.2f}'.format(Acc * 100))
        print('Original avg conf {0:.2f}'.format(list_softmax.mean()))
        print('MFS avg conf {0:.2f}'.format(list_softmax_mfs.mean()))

