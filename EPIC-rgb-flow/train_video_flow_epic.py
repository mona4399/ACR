from mmaction.apis import init_recognizer
import torch
import argparse
import tqdm
import os
import numpy as np
import torch.nn as nn
import random
from dataloader_video_flow_epic import EPICDOMAIN
import torch.nn.functional as F

def train_one_step(model, clip, labels, flow, model_flow, num_classes, index):
    clip = clip['imgs'].cuda().squeeze(1)
    labels = labels.cuda()
    flow = flow['imgs'].cuda().squeeze(1)

    with torch.no_grad():
        audio_feat = model_flow.module.backbone.get_feature(flow)
        x_slow, x_fast = model.module.backbone.get_feature(clip) 
        v_feat = (x_slow.detach(), x_fast.detach())  
        
    v_feat = model.module.backbone.get_predict(v_feat)
    v_predict, v_emd = model.module.cls_head(v_feat)

    audio_feat = model_flow.module.backbone.get_predict(audio_feat.detach())
    f_predict, f_emd = model_flow.module.cls_head(audio_feat)

    predict = mlp_cls(v_emd, f_emd)

    if args.use_single_pred:
        loss = (criterion(predict, labels) + criterion(v_predict, labels) + criterion(f_predict, labels)) / 3
    else:
        loss = criterion(predict, labels)

    if args.use_acl:
        v_soft = F.softmax(v_predict, dim=1)
        f_soft = F.softmax(f_predict, dim=1)
        fused_soft = F.softmax(predict, dim=1)

        v_conf, v_pred = torch.max(v_soft, dim=1)
        f_conf, f_pred = torch.max(f_soft, dim=1)
        fused_conf, fused_pred = torch.max(fused_soft, dim=1)

        confidence_penalty_loss = (torch.clamp(v_conf - fused_conf, min=0).mean() + torch.clamp(f_conf - fused_conf, min=0).mean()) / 2

        loss = loss + args.acl_loss * confidence_penalty_loss

    if args.use_mfs:
        num_mixing = random.randint(args.mfs_min, args.mfs_max)

        v_start = torch.randint(0, v_dim - num_mixing + 1, (1,)).item()
        f_start = torch.randint(0, f_dim - num_mixing + 1, (1,)).item()

        v_emd_new = v_emd.clone()
        f_emd_new = f_emd.clone()

        v_block = v_emd[:, v_start:v_start+num_mixing].clone()
        f_block = f_emd[:, f_start:f_start+num_mixing].clone()

        v_emd_new[:, v_start:v_start+num_mixing] = f_block
        f_emd_new[:, f_start:f_start+num_mixing] = v_block
        predict_fm = mlp_cls(v_emd_new, f_emd_new)

        labels_one_hot = torch.nn.functional.one_hot(labels, num_classes=num_classes).float()
        labels_extra = torch.full_like(labels, fill_value=num_classes-1).cuda()
        labels_extra_one_hot = torch.nn.functional.one_hot(labels_extra, num_classes=num_classes).float()
        #lamda = num_mixing / f_dim
        lamda = num_mixing / args.mfs_max
        mixed_labels = (1-lamda) * labels_one_hot + lamda * labels_extra_one_hot

        loss = loss + args.mfs_loss * criterion(predict_fm, mixed_labels)

    optim.zero_grad()
    loss.backward()
    optim.step()
    return predict, loss

def validate_one_step(model, clip, labels, flow, model_flow):
    clip = clip['imgs'].cuda().squeeze(1)
    labels = labels.cuda()
    flow = flow['imgs'].cuda().squeeze(1)

    with torch.no_grad():
        x_slow, x_fast = model.module.backbone.get_feature(clip) 
        v_feat = (x_slow.detach(), x_fast.detach())  

        v_feat = model.module.backbone.get_predict(v_feat)
        v_predict, v_emd = model.module.cls_head(v_feat)

        audio_feat = model_flow.module.backbone.get_feature(flow)  
        audio_feat = model_flow.module.backbone.get_predict(audio_feat)
        f_predict, f_emd = model_flow.module.cls_head(audio_feat)
       
        predict = mlp_cls(v_emd, f_emd)

        if args.use_mfs:
            predict = predict[:, :-1]


    loss = criterion(predict, labels)

    return predict, loss


class Encoder(nn.Module):
    def __init__(self, input_dim=2816, out_dim=8):
        super(Encoder, self).__init__()
        self.enc_net = nn.Linear(input_dim, out_dim)
       
    def forward(self, vfeat, afeat):
        feat = torch.cat((vfeat, afeat), dim=1)
        return self.enc_net(feat)

class ProjectHead(nn.Module):
    def __init__(self, input_dim=2816, hidden_dim=2048, out_dim=128):
        super(ProjectHead, self).__init__()
        self.head = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, out_dim)
            )
        
    def forward(self, feat):
        feat = F.normalize(self.head(feat), dim=1)
        return feat
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--datapath', type=str, default='/path/to/video_datasets/',
                        help='datapath')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='lr')
    parser.add_argument('--bsz', type=int, default=16,
                        help='batch_size')
    parser.add_argument("--nepochs", type=int, default=50)
    parser.add_argument('--save_checkpoint', action='store_true')
    parser.add_argument('--save_best', action='store_true')
    parser.add_argument("--opt", type=str, default='adam')
    parser.add_argument('--resumef', action='store_true')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--appen", type=str, default='')
    parser.add_argument('--use_single_pred', action='store_true')
    parser.add_argument('--use_acl', action='store_true')
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--dataset", type=str, default='EPIC')

    parser.add_argument('--mfs_loss', type=float, default=1.0,
                        help='mfs_loss')
    parser.add_argument('--mfs_min', type=int, default=32,
                        help='mfs_min')
    parser.add_argument('--mfs_max', type=int, default=512,
                        help='mfs_max')

    parser.add_argument('--acl_loss', type=float, default=2.0,
                        help='acl_loss')
    parser.add_argument('--use_mfs', action='store_true')

    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # init_distributed_mode(args)
    config_file = 'configs/recognition/slowfast/slowfast_r101_8x8x1_256e_kinetics400_rgb.py'
    checkpoint_file = 'pretrained_models/slowfast_r101_8x8x1_256e_kinetics400_rgb_20210218-0dd54025.pth'

    config_file_flow = 'configs/recognition/slowonly/slowonly_r50_8x8x1_256e_kinetics400_flow.py'
    checkpoint_file_flow = 'pretrained_models/slowonly_r50_8x8x1_256e_kinetics400_flow_20200704-6b384243.pth'

    # assign the desired device.
    device = 'cuda:0' # or 'cpu'
    device = torch.device(device)

    v_dim = 2304
    f_dim = 2048

    
    num_class = 8

    if args.use_mfs:
        num_class = num_class + 1

    # build the model from a config file and a checkpoint file
    model = init_recognizer(config_file, checkpoint_file, device=device, use_frames=True)
    model.cls_head.fc_cls = nn.Linear(v_dim, num_class).cuda()
    cfg = model.cfg
    model = torch.nn.DataParallel(model)

    model_flow = init_recognizer(config_file_flow, checkpoint_file_flow, device=device,use_frames=True)
    model_flow.cls_head.fc_cls = nn.Linear(f_dim, num_class).cuda()
    cfg_flow = model_flow.cfg
    model_flow = torch.nn.DataParallel(model_flow)

    mlp_cls = Encoder(input_dim=v_dim+f_dim, out_dim=num_class)
    mlp_cls = mlp_cls.cuda()

    base_path = "checkpoints/"
    if not os.path.exists(base_path):
        os.mkdir(base_path)
    base_path_model = "models/"
    if not os.path.exists(base_path_model):
        os.mkdir(base_path_model)

    log_name = "log_video_flow_%s_lr_%s_bsz_%s_%s_%s"%(str(args.dataset), str(args.lr), str(args.bsz), str(args.nepochs), args.opt)

    if args.use_single_pred:
        log_name = log_name + '_single_pred'
    if args.use_acl:
        log_name = log_name + '_ada_conf_diff_%s'%(str(args.acl_loss))
    if args.use_mfs:
        log_name = log_name + '_cutmix_%s_%s_%s'%(str(args.mfs_loss), str(args.mfs_min), str(args.mfs_max))

    log_name = log_name + args.appen
    log_path = base_path + log_name + '.csv'
    print(log_path)
    
    criterion = nn.CrossEntropyLoss()

    criterion = criterion.cuda()
    batch_size = args.bsz

    params = list(model.module.backbone.fast_path.layer4.parameters()) + list(
        model.module.backbone.slow_path.layer4.parameters()) +list(model.module.cls_head.parameters())+list(model_flow.module.backbone.layer4.parameters()) +list(model_flow.module.cls_head.parameters())
    params = params + list(mlp_cls.parameters())

    if args.opt == 'adam':
        optim = torch.optim.Adam(params, lr=args.lr, weight_decay=1e-4)
    elif args.opt == 'sgd':
        optim = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)

    BestLoss = float("inf")
    BestEpoch = 0
    BestAcc = 0
    BestTestAcc = 0

    if args.resumef:
        resume_file = base_path_model + log_name + '.pt'
        print("Resuming from ", resume_file)
        checkpoint = torch.load(resume_file)
        starting_epoch = checkpoint['epoch']+1
    
        BestLoss = checkpoint['BestLoss']
        BestEpoch = checkpoint['BestEpoch']
        BestAcc = checkpoint['BestAcc']

        model.load_state_dict(checkpoint['model_state_dict'])
        model_flow.load_state_dict(checkpoint['model_flow_state_dict'])
        optim.load_state_dict(checkpoint['optimizer'])
        mlp_cls.load_state_dict(checkpoint['mlp_cls_state_dict'])
    else:
        print("Training From Scratch ..." )
        starting_epoch = 0

    print("starting_epoch: ", starting_epoch)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = EPICDOMAIN(split='train', cfg=cfg, cfg_flow=cfg_flow, datapath=args.datapath)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=args.num_workers, shuffle=True,
                                                   pin_memory=(device.type == "cuda"), drop_last=True)

    val_dataset = EPICDOMAIN(split='val', cfg=cfg, cfg_flow=cfg_flow, datapath=args.datapath)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, num_workers=args.num_workers, shuffle=False,
                                                   pin_memory=(device.type == "cuda"), drop_last=False)
    dataloaders = {'train': train_dataloader, 'val': val_dataloader}
 
    with open(log_path, "a") as f:
        for epoch_i in range(starting_epoch, args.nepochs):
            print("Epoch: %02d" % epoch_i)
            for split in ['train', 'val']:
                acc = 0
                count = 0
                total_loss = 0
                print(split)
                model.train(split == 'train')
                model_flow.train(split == 'train')
                mlp_cls.train(split == 'train')

                with tqdm.tqdm(total=len(dataloaders[split])) as pbar:
                    for (i, (clip, spectrogram, labels, index)) in enumerate(dataloaders[split]):
                        if split=='train':
                            predict1, loss = train_one_step(model, clip, labels, spectrogram, model_flow, num_class, index)
                        else:
                            predict1, loss = validate_one_step(model, clip, labels, spectrogram, model_flow)

                        total_loss += loss.item() * batch_size
                        _, predict = torch.max(predict1.detach().cpu(), dim=1)

                        acc1 = (predict == labels).sum().item()
                        acc += int(acc1)
                        count += predict1.size()[0]
                        pbar.set_postfix_str(
                            "Average loss: {:.4f}, Current loss: {:.4f}, Accuracy: {:.4f}".format(total_loss / float(count),
                                                                                                  loss.item(),
                                                                                                  acc / float(count)))
                        pbar.update()

                    if split == 'val':
                        currentvalAcc = acc / float(count)
                        if currentvalAcc >= BestAcc:
                            BestLoss = total_loss / float(count)
                            BestEpoch = epoch_i
                            BestAcc = acc / float(count)

                            if args.save_best:
                                save = {
                                    'epoch': epoch_i,
                                    'BestLoss': BestLoss,
                                    'BestEpoch': BestEpoch,
                                    'BestAcc': BestAcc,
                                    'model_state_dict': model.state_dict(),
                                    'model_flow_state_dict': model_flow.state_dict(),
                                    # 'optimizer': optim.state_dict(),
                                }
                                save['mlp_cls_state_dict'] = mlp_cls.state_dict()

                                torch.save(save, base_path_model + log_name + '_best.pt')
                            

                    if args.save_checkpoint:
                        save = {
                            'epoch': epoch_i,
                            'BestLoss': BestLoss,
                            'BestEpoch': BestEpoch,
                            'BestAcc': BestAcc,
                            'model_state_dict': model.state_dict(),
                            'model_flow_state_dict': model_flow.state_dict(),
                            # 'optimizer': optim.state_dict(),
                        }
                        save['mlp_cls_state_dict'] = mlp_cls.state_dict()
                        torch.save(save, base_path_model + log_name + '.pt')
                        
                    f.write("{},{},{},{}\n".format(epoch_i, split, total_loss / float(count), acc / float(count)))
                    f.flush()

                    print('acc on epoch ', epoch_i)
                    print("{},{},{}\n".format(epoch_i, split, acc / float(count)))
                    print('BestValAcc ', BestAcc)
                    
                    if split == 'val':
                        f.write("CurrentBestEpoch,{},BestLoss,{},BestValAcc,{} \n".format(BestEpoch, BestLoss, BestAcc))
                        f.flush()

        f.write("BestEpoch,{},BestLoss,{},BestValAcc,{} \n".format(BestEpoch, BestLoss, BestAcc))
        f.flush()

        print('BestValAcc ', BestAcc)

    f.close()

