"""
Builds upon: https://github.com/tim-learn/SHOT
Corresponding paper: http://proceedings.mlr.press/v119/liang20a/liang20a.pdf
"""

import os.path as osp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from src.utils import loss
from src.models import network
from torch.utils.data import DataLoader
from src.data.data_list import ImageList, ImageList_idx
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix
from src.utils.utils import *

from .source import ImageScaling, squeeze_first
from kornia.augmentation import RandomHorizontalFlip, RandomAffine, ColorJitter
from FLAIR import flair_main
from src.utils import IID_losses

import matplotlib.pyplot as plt
import cv2
from matplotlib.colors import ListedColormap
from FLAIR.flair.modeling.misc import set_seeds
from iopath.common.file_io import g_pathmgr

logger = logging.getLogger(__name__)

def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer

def lr_scheduler(cfg,optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = cfg.OPTIM.WD
        param_group['momentum'] = cfg.OPTIM.MOMENTUM
        param_group['nesterov'] = cfg.OPTIM.NESTEROV
    return optimizer

def image_train(resize_size=256, crop_size=224, alexnet=False, fundus=False):
  if not alexnet:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
#   else:
#     normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
  if fundus:
      return  transforms.Compose([
        transforms.ToTensor(),
        ImageScaling(size=(512, 512)),
        RandomHorizontalFlip(p=0.5),
        RandomAffine(p=0.25, degrees=(-5, 5), scale=(0.9, 1)),
        squeeze_first()
    ])
  
  return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

def image_test(resize_size=256, crop_size=224, alexnet=False, fundus=False):
  if not alexnet:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
#   else:
#     normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
  if fundus:
      return  transforms.Compose([
        transforms.ToTensor(),
        ImageScaling(size=(512, 512))
    ])
    
  return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize
    ])

def data_load(cfg): 
    ## prepare data
    dsets = {}
    dset_loaders = {}
    train_bs = cfg.TEST.BATCH_SIZE
    txt_tar = open(cfg.t_dset_path).readlines()
    txt_test = open(cfg.t_dset_path).readlines()

    # if not cfg.da == 'uda':
    #     label_map_s = {}
    #     for i in range(len(cfg.src_classes)):
    #         label_map_s[cfg.src_classes[i]] = i

    #     new_tar = []
    #     for i in range(len(txt_tar)):
    #         rec = txt_tar[i]
    #         reci = rec.strip().split(' ')
    #         if int(reci[1]) in cfg.tar_classes:
    #             if int(reci[1]) in cfg.src_classes:
    #                 line = reci[0] + ' ' + str(label_map_s[int(reci[1])]) + '\n'   
    #                 new_tar.append(line)
    #             else:
    #                 line = reci[0] + ' ' + str(len(label_map_s)) + '\n'   
    #                 new_tar.append(line)
    #     txt_tar = new_tar.copy()
    #     txt_test = txt_tar.copy()
    fundus = 'fundus' in cfg.SETTING.DATASET

    dsets["target"] = ImageList_idx(txt_tar, transform=image_train(fundus=fundus))
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, shuffle=True, num_workers=cfg.NUM_WORKERS, drop_last=False)
    dsets["test"] = ImageList_idx(txt_test, transform=image_test(fundus=fundus))
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs*3, shuffle=False, num_workers=cfg.NUM_WORKERS, drop_last=False)

    return dset_loaders

def cal_acc(loader, netF, netB, netC, flag=False, iter_num=-1, return_pred=False):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            outputs = netC(netB(netF(inputs)))
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(loss.Entropy(nn.Softmax(dim=1)(all_output))).cpu().data.item()
    
    if iter_num != -1:
        prob = nn.Softmax(dim=1)(all_output).cpu()


    if flag:
        matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        acc = matrix.diagonal()/matrix.sum(axis=1) * 100
        aacc = acc.mean()
        aa = [str(np.round(i, 2)) for i in acc]
        acc = ' '.join(aa)
        if return_pred:
            prob = prob.cuda()
            predict = predict.cuda()
            return aacc, acc, prob, predict
        return aacc, acc
    else:
        return accuracy*100, mean_ent

def cal_acc_ensemble(loader, netF, netB, netC, adapter, flag=False):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            outputs = netC(netB(netF(inputs)))
            preds_clip = adapter.predict_batch(inputs)
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
                all_preds_clip = preds_clip.float().cpu()
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
                all_preds_clip = torch.cat((all_preds_clip, preds_clip.float().cpu()), 0)
    all_output = nn.Softmax(dim=1)(all_output)
    #_, predict = torch.max((all_output+all_preds_clip)/2, 1)
    _, predict = torch.max(all_preds_clip, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(loss.Entropy(nn.Softmax(dim=1)(all_output))).cpu().data.item()
    
    """ ###vis confidence distribution
    prob = all_preds_clip
    save_path = f'./src_confidence/fundus_4c/0.png'
    plot_confi_distri(prob, all_label, save_path) """
    
    if flag:
        matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        acc = matrix.diagonal()/matrix.sum(axis=1) * 100
        aacc = acc.mean()
        aa = [str(np.round(i, 2)) for i in acc]
        acc = ' '.join(aa)
        return aacc, acc
    else:
        return accuracy*100, mean_ent

def visualize_patch_preds(image, clip_patch_pred, clip_patch_prob, target_patch_pred, target_patch_prob, output_path):
    colors = ['red', 'green', 'violet', 'cyan']
    cmap = ListedColormap(colors)
    
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    axes[0].set_title("Original Image")
    axes[0].imshow(image)
    axes[0].axis("off")

    axes[1].set_title("clip_patch_pred")
    axes[1].imshow(clip_patch_pred, cmap=cmap, vmin=0, vmax=3)
    for i in range(clip_patch_prob.shape[0]):
        for j in range(clip_patch_prob.shape[1]):
            axes[1].text(j, i, f"{clip_patch_prob[i, j]:.2f}",  # 格式化为两位小数
                        ha='center', va='center', color='black', fontsize=7)
    
    axes[2].set_title("target_patch_pred")
    im2 = axes[2].imshow(target_patch_pred, cmap=cmap, vmin=0, vmax=3)
    for i in range(target_patch_prob.shape[0]):
        for j in range(target_patch_prob.shape[1]):
            axes[2].text(j, i, f"{target_patch_prob[i, j]:.2f}",  # 格式化为两位小数
                        ha='center', va='center', color='black', fontsize=7)


    cbar = fig.colorbar(im2, ax=axes[1:], orientation='vertical', fraction=0.02, pad=0.01, ticks=[0, 1, 2, 3])
    cbar.ax.set_yticklabels(['0', '1', '2', '3'])  # 自定义标签

    plt.tight_layout(rect=(0,0,0.89,1))
    plt.savefig(output_path)


def plot_patch_preds(loader, netF, netB, netC, adapter, save_path):
    netF.eval()
    netB.eval()
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            indices = data[2]
            inputs = inputs.cuda()

            fea_F, layer4_fea = netF(inputs, return_last_fea=True)
            features_test = netB(fea_F)
            fea_proj = netB(layer4_fea.permute(0,2,3,1))
            outputs_test = netC(features_test)
            outputs_patch = netC(fea_proj)

            softmax_out = nn.Softmax(dim=1)(outputs_test)
            _, predicts = torch.max(softmax_out, 1)
            softmax_patch = nn.Softmax(dim=-1)(outputs_patch)
            target_patch_prob, target_patch_preds = torch.max(softmax_patch, -1)


            clip_preds, clip_patch_preds = adapter.predict_batch(inputs, return_cam=True)
            _, clip_preds = torch.max(clip_preds, 1)
            clip_patch_prob, clip_patch_preds = torch.max(clip_patch_preds, -1)

            predicts = predicts.cpu().numpy()
            target_patch_preds = target_patch_preds.cpu().numpy()
            target_patch_prob = target_patch_prob.cpu().numpy()
            clip_preds = clip_preds.cpu().numpy()
            clip_patch_preds = clip_patch_preds.cpu().numpy()
            clip_patch_prob = clip_patch_prob.cpu().numpy()
            for i, ind in enumerate(indices):
                path = loader.dataset.imgs[ind][0]
                image = loader.dataset.loader(path)
                save_name = path.split('/')[-1].split('_')[0]
                save_name = save_name + '_gt_{:d}_flair_{:d}_src_{:d}.png'.format(labels[i].item(), clip_preds[i], predicts[i])

                visualize_patch_preds(image, clip_patch_preds[i], clip_patch_prob[i], target_patch_preds[i], target_patch_prob[i], output_path=osp.join(save_path, save_name))
                
    netF.train()
    netB.train()        


    
def train_target(cfg):
    dset_loaders = data_load(cfg)
    ## set base network
    if cfg.MODEL.ARCH[0:3] == 'res':
        netF = network.ResBase(res_name=cfg.MODEL.ARCH).cuda()
    elif cfg.MODEL.ARCH[0:3] == 'vgg':
        netF = network.VGGBase(vgg_name=cfg.MODEL.ARCH).cuda()  

    netB = network.feat_bottleneck(type='ori', feature_dim=netF.in_features, bottleneck_dim=cfg.bottleneck).cuda()###type='bn'
    netC = network.feat_classifier(type='wn', class_num = cfg.class_num, bottleneck_dim=cfg.bottleneck).cuda()

    modelpath = cfg.output_dir_src + '/source_F.pt'   
    netF.load_state_dict(torch.load(modelpath))
    modelpath = cfg.output_dir_src + '/source_B.pt'   
    netB.load_state_dict(torch.load(modelpath))
    modelpath = cfg.output_dir_src + '/source_C.pt'    
    netC.load_state_dict(torch.load(modelpath))
    netC.eval()
    for k, v in netC.named_parameters():
        v.requires_grad = False

    """ print('source model:')
    print(netC.fc.weight_v.size(),netC.fc.weight_g.size())
    weight = netC.fc.weight_v/netC.fc.weight_v.norm(dim=-1,keepdim=True)
    print(torch.matmul(weight,weight.transpose(0,1))) """

    param_group = []
    for k, v in netF.named_parameters():
        if cfg.OPTIM.LR_DECAY1 > 0:
            param_group += [{'params': v, 'lr': cfg.OPTIM.LR * cfg.OPTIM.LR_DECAY1}]
        else:
            v.requires_grad = False
    for k, v in netB.named_parameters():
        if cfg.OPTIM.LR_DECAY2 > 0:
            param_group += [{'params': v, 'lr': cfg.OPTIM.LR * cfg.OPTIM.LR_DECAY2}]
        else:
            v.requires_grad = False

    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    max_iter = cfg.TEST.MAX_EPOCH * len(dset_loaders["target"])
    interval_iter = max_iter // cfg.TEST.INTERVAL
    iter_num = 0

    adapter = flair_main.main(cfg.SETTING.DATASET, cfg.SETTING.T)

    #all_preds = adapter.predict_image(dset_loaders["test"])
    #exit(0)
    set_seeds(42, use_cuda=torch.cuda.is_available())#42

    """ netF.eval()
    netB.eval()
    acc_s_te, acc_list = cal_acc_ensemble(dset_loaders['test'], netF, netB, netC, adapter, True)
    log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(cfg.name, iter_num, max_iter, acc_s_te) + '\n' + acc_list
    logging.info(log_str) """
    
    
    netF.eval()
    netB.eval()
    acc_s_te, acc_list, src_prob, src_predict = cal_acc(dset_loaders['test'], netF, netB, netC, True, iter_num=iter_num, return_pred=True)
    print('source:')
    print(acc_s_te, acc_list)
    
    
    netF.train()
    netB.train()
    while iter_num < max_iter:
        try:
            inputs_test, _, tar_idx = next(iter_test)
        except:
            iter_test = iter(dset_loaders["target"])
            inputs_test, _, tar_idx = next(iter_test)
        

        if inputs_test.size(0) == 1:
            continue

        inputs_test = inputs_test.cuda()
        preds, patch_preds = adapter.predict_batch(inputs_test, return_cam=True)
        #preds = all_preds[tar_idx]

        iter_num += 1
        lr_scheduler(cfg,optimizer, iter_num=iter_num, max_iter=max_iter)

        fea_F, layer4_fea = netF(inputs_test, return_last_fea=True)
        features_test = netB(fea_F)
        fea_proj = netB(layer4_fea.permute(0,2,3,1))
        outputs_test = netC(features_test)
        outputs_patch = netC(fea_proj)

        softmax_out = nn.Softmax(dim=1)(outputs_test)
        softmax_patch = nn.Softmax(dim=-1)(outputs_patch)

        classifier_loss = 1.0*IID_losses.IID_loss(softmax_out, preds)

        src_mask = torch.max(src_prob[tar_idx], dim=1)[0] > 0.95 
        classifier_loss_src = IID_losses.IID_loss(softmax_out[src_mask], src_prob[tar_idx][src_mask],lamb=cfg.PATCH.lamb_src)
        classifier_loss += 1.0*classifier_loss_src

        softmax_patch = softmax_patch[:,2:-2,2:-2,:]
        patch_preds = patch_preds[:,2:-2,2:-2,:]

        #remove those incompatible with src preds
        if src_mask.sum().item() > 0:
            src_label = torch.argmax(src_prob[tar_idx], dim=1)
            src_compat_indice = torch.argmax(patch_preds[src_mask], dim=-1) == src_label[src_mask].unsqueeze(-1).unsqueeze(-1)
            patch_preds_src = patch_preds[src_mask][src_compat_indice]
            softmax_patch_src = softmax_patch[src_mask][src_compat_indice]
            #print(softmax_patch.shape,src_compat_indice.shape,softmax_patch_src.shape)
            
            patch_preds_nosrc = patch_preds[torch.logical_not(src_mask)]
            softmax_patch_nosrc = softmax_patch[torch.logical_not(src_mask)]
            patch_preds_nosrc = patch_preds_nosrc.contiguous().view(-1, patch_preds_nosrc.size(-1))
            softmax_patch_nosrc = softmax_patch_nosrc.contiguous().view(-1, softmax_patch_nosrc.size(-1))
            patch_preds = torch.cat((patch_preds_src, patch_preds_nosrc), dim=0)
            softmax_patch = torch.cat((softmax_patch_src, softmax_patch_nosrc), dim=0)
            #patch_preds = patch_preds_src
            #softmax_patch = softmax_patch_src

        else:
            softmax_patch = softmax_patch.contiguous().view(-1, softmax_patch.size(-1))
            patch_preds = patch_preds.contiguous().view(-1, patch_preds.size(-1))

        patch_select = patch_preds.max(dim=1)[0] > 0.95
        patch_select = patch_select.detach()

        softmax_patch, patch_preds = balance_process(softmax_patch, patch_preds, patch_select)
        
        classifier_loss_patch = IID_losses.IID_loss(softmax_patch, patch_preds)
        classifier_loss += 0.3*max((max_iter/2-iter_num)/(max_iter/2), 0)*classifier_loss_patch #max((max_iter/2-iter_num)/(max_iter/2), 0)


        classifier_loss = classifier_loss

    
        """ if cfg.SHOT.CLS_PAR > 0:
            pred = mem_label[tar_idx]
            classifier_loss = nn.CrossEntropyLoss()(outputs_test, pred)
            classifier_loss *= cfg.SHOT.CLS_PAR
            if iter_num < interval_iter and cfg.SETTING.DATASET == "VISDA-C":
                classifier_loss *= 0
        else:
            classifier_loss = torch.tensor(0.0).cuda() """

        #entropy_loss = torch.mean(loss.Entropy(softmax_out))
        msoftmax = softmax_out.mean(dim=0)
        gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + cfg.LCFD.EPSILON))
        classifier_loss = classifier_loss - cfg.PATCH.GENT_PAR * gentropy_loss

        """ selected_preds, selected_labels = ohem_select(outputs_patch[:,2:-2,2:-2,:].contiguous(), patch_preds[:,2:-2,2:-2,:].contiguous(), threshold=0.99, top_k=0.15)
        labels = selected_labels.argmax(dim=1)
        stat = [torch.sum(labels == c).item() for c in range(4)]
        logging.info(stat)
        ohem_loss = F.cross_entropy(selected_preds, selected_labels.argmax(dim=1))
        classifier_loss += 0.01*ohem_loss """


        optimizer.zero_grad()
        classifier_loss.backward()
        optimizer.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netF.eval()
            netB.eval()
            if cfg.SETTING.DATASET=='VISDA-C' or 'fundus' in cfg.SETTING.DATASET:
                acc_s_te, acc_list, src_prob, src_predict = cal_acc(dset_loaders['test'], netF, netB, netC, True, iter_num=iter_num, return_pred=True)
                log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(cfg.name, iter_num, max_iter, acc_s_te) + '\n' + acc_list
            else:
                acc_s_te, _ = cal_acc(dset_loaders['test'], netF, netB, netC, False)
                log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(cfg.name, iter_num, max_iter, acc_s_te)

            logging.info(log_str)
            netF.train()
            netB.train()

            #if iter_num==650:
            #    break

    """ plot_patch_preds(dset_loaders["test"], netF, netB, netC, adapter, save_path=osp.join('./vis_patch_preds', cfg.SETTING.DATASET, 'after_adapt'))
    exit(0) """
    
    #print(iter_num)
    #cfg.output_dir = os.path.join('./saved_model',cfg.SETTING.DATASET,cfg.domain[cfg.SETTING.S][0].upper()+cfg.domain[cfg.SETTING.T][0].upper(),cfg.MODEL.METHOD)
    #cfg.output_dir = os.path.join('./saved_model',cfg.SETTING.DATASET,cfg.domain[cfg.SETTING.S][0].upper()+cfg.domain[cfg.SETTING.T][0].upper(),'patch_wo_la')
    #g_pathmgr.mkdirs(cfg.output_dir)
    cfg.ISSAVE = False
    if cfg.ISSAVE:   
        torch.save(netF.state_dict(), osp.join(cfg.output_dir, "target_F_" + str(iter_num) + ".pt"))
        torch.save(netB.state_dict(), osp.join(cfg.output_dir, "target_B_" + str(iter_num) + ".pt"))
        torch.save(netC.state_dict(), osp.join(cfg.output_dir, "target_C_" + str(iter_num) + ".pt"))
        
    return netF, netB, netC

def print_cfg(cfg):
    s = "==========================================\n"
    for arg, content in cfg.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s

def ohem_select(patch_predictions, patch_labels, threshold=0.99, top_k=0.1):
    
    B, H, W, num_classes = patch_predictions.shape
    with torch.no_grad():
        losses = F.cross_entropy(patch_predictions.view(-1, num_classes), patch_labels.argmax(dim=3).view(-1), reduction='none')
    #losses = losses.view(B*H*W)

    # 筛选可靠标签
    # 判断标签中最大概率是否大于阈值
    max_probs = patch_labels.max(dim=3)[0].view(-1)
    reliable_mask = max_probs > threshold
    
    num_reliable = torch.sum(reliable_mask)
    top_k = int(num_reliable.item()*top_k)

    # 筛选 hard example
    losses[~reliable_mask] = -1  # 将不可靠的样本排除
    _, hard_indices = torch.topk(losses, top_k, dim=0)  # 按照损失排序，取 top_k
    
    # 根据 indices 选出对应的 patch、预测和标签
    selected_preds = patch_predictions.view(-1, num_classes)[hard_indices]
    selected_labels = patch_labels.view(-1, num_classes)[hard_indices]


    return selected_preds, selected_labels

def balance_process(softmax_patch, patch_preds, patch_select):
    softmax_patch = softmax_patch[patch_select]
    patch_preds = patch_preds[patch_select]
    patch_label = patch_preds.max(dim=1)[1]
    
    cls_num = patch_preds.size(-1)
    class_counts = [(patch_label==i).sum().item() for i in range(cls_num)]
    class_counts = torch.tensor(class_counts).cuda()

    sample_weights = class_counts[patch_label]

    #对每个样本的概率向量除以对应的类别出现次数
    weighted_patch_preds = patch_preds / sample_weights.unsqueeze(1)

    return softmax_patch, weighted_patch_preds

