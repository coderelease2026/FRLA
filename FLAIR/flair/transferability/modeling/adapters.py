"""
This script contains adapters for fast adaptation of
FLAIR modelo to downstream tasks/domains.

In particular, these adapters work over the vision and text
embeddings. Also, this code contains a Wrapper for zero-shot
classification

Implemented adapters:
Zero-shot, Linear Probe (LP), ClipAdapter, TipAdapter, TipAdapter-f
"""

import copy
import random
import torch
import numpy as np

from tqdm import tqdm
from sklearn.linear_model import LogisticRegression

from FLAIR.flair.pretraining.data.transforms import augmentations_pretraining

from src.utils import vis
from scipy.spatial.distance import cdist
from FLAIR.flair.utils.metrics import evaluate
from src.utils import loss
# Device for training/inference
device = 'cuda' if torch.cuda.is_available() else 'cpu'

"""
The first section contains only-vision adapters (i.e. linear probing)
"""


class AdapterWrapper(object):
    def __init__(self, model, targets, tta=False, fta=False):
        # Set model and number of targets
        self.model = copy.deepcopy(model)
        self.model.eval()
        self.num_targets = len(targets)
        # Augmentation for training and for test-time augmentation
        self.tta = tta
        self.fta = fta
        self.number_augmentations = 20

    def extract_vision_features(self, data_loader, transforms=None, return_last_fea=False):
        self.model.eval()

        epoch_iterator = tqdm(
            data_loader, desc="Extracting features (X / X Steps)", dynamic_ncols=True
        )

        X, Y = [], []
        Z = []
        
        for step, batch in enumerate(epoch_iterator):
            ###images = batch["image"].to(device).to(torch.float32)
            images = batch[0].to(device).to(torch.float32)

            with torch.no_grad():

                # Image augmentation
                if transforms is not None:
                    images = transforms(images)

                # Forward vision encoder
                if return_last_fea:
                    x, project_fea = self.model.vision_model(images, return_last_fea=True)
                else:
                    x = self.model.vision_model(images)

            X.extend(x.cpu().detach().numpy())
            ###Y.extend(batch["label"].numpy())
            Y.extend(batch[1].numpy())
            if return_last_fea:
                Z.append(project_fea)
        
        X = np.array(X)
        Y = np.array(Y)
        if return_last_fea:
            last_fea = torch.concat(Z)
            return X, Y, last_fea
        return X, Y
    
    def extract_vision_features_batch(self, images, transforms=None, return_last_fea=False):
        self.model.eval()
        
        images = images.to(torch.float32)

        with torch.no_grad():

            # Image augmentation
            if transforms is not None:
                images = transforms(images)

            # Forward vision encoder
            if return_last_fea:
                x, project_fea = self.model.vision_model(images, return_last_fea=True)
            else:
                x = self.model.vision_model(images)

        if return_last_fea:
            return x, project_fea
        return x

    def fit(self, loaders, transforms=None):
        data_loader = loaders["train"]

        if self.fta:
            transforms = augmentations_pretraining

        # Extract features and labels from generator
        if self.fta and transforms is not None:
            X, Y = [], []
            for i in range(self.number_augmentations):
                Xa, Ya = self.extract_vision_features(data_loader, transforms=transforms)
                X.append(Xa), Y.append(Ya)
            X = np.concatenate(X, 0)
            Y = np.concatenate(Y, 0)
        else:
            X, Y = self.extract_vision_features(data_loader, transforms=transforms)

        # Perform logistic regression
        self.train(X, Y)

    def train(self, X, Y):
        """
        Placeholder: function to be developed in a concrete adapter.
        """
        return

    def predict(self, loader, transforms=None):
        """
        Placeholder: function to be developed in a concrete adapter.
        """
        return


class LinearProbe(AdapterWrapper):
    def __init__(self, model, targets, tta=False, fta=False, c=0.316):
        super().__init__(model, targets, tta=tta, fta=fta)
        self.classifier = LogisticRegression(random_state=0, C=c, max_iter=1000, verbose=0,
                                             class_weight="balanced")

    def train(self, X, Y):

        # Train classifier
        self.classifier.fit(X, Y)

        # Set Linear Probe classifier into FLAIR model
        self.model.classifier = torch.nn.Linear(X.shape[-1], self.num_targets, bias=True)
        self.model.classifier.weight = torch.nn.Parameter(torch.tensor(self.classifier.coef_).to(torch.float32))
        self.model.classifier.bias = torch.nn.Parameter(torch.tensor(self.classifier.intercept_).to(torch.float32))
        self.model.classifier.to(device)

    def predict(self, loader, transforms=None):
        self.model.eval()

        # Set transforms on test-time augmentation
        if self.tta:
            transforms = augmentations_pretraining

        epoch_iterator = tqdm(
            loader, desc="Predicting (X / X Steps) (loss=X.X)", dynamic_ncols=True
        )

        with torch.no_grad():
            refs, preds = [], []
            for step, batch in enumerate(epoch_iterator):
                images = batch["image"].to(device).to(torch.float32)
                Y = batch["label"].to(device).to(torch.long)

                # Forward
                if self.tta:
                    preds_tta = []
                    for i in range(self.number_augmentations):
                        x = self.model.vision_model(transforms(images))
                        score = self.model.classifier(x)
                        preds_tta.append(score.unsqueeze(-1))
                    score = torch.concat(preds_tta, -1).mean(-1)
                else:
                    x = self.model.vision_model(images)
                    score = self.model.classifier(x)
                # Activation for prediction
                if score.shape[-1] == 1:  # Binary case
                    score = torch.sigmoid(score)
                    score = torch.concat([1 - score, score], -1)
                else:  # Multi-class case
                    score = torch.softmax(score, -1)
                torch.cuda.empty_cache()

                refs.append(Y.cpu().detach().numpy())
                preds.append(score.cpu().detach().numpy())

        refs = np.concatenate(refs, 0)
        preds = np.concatenate(preds, 0)
        return refs, preds


"""
This section contains multimodal (vision-language) adapters.
"""


class LanguageAdapterWrapper(AdapterWrapper):
    def __init__(self, model, targets, domain_knowledge=False, tta=False, fta=False):
        super().__init__(model, targets, tta=tta, fta=fta)

        # Compute text prototypes
        self.text_embeds_dict, self.text_embeds = model.compute_text_embeddings(list(targets.keys()),
                                                                                domain_knowledge=domain_knowledge)


class ZeroShot(LanguageAdapterWrapper):
    def __init__(self, model, targets, domain_knowledge=False, tta=False, fta=False):
        super().__init__(model, targets, domain_knowledge=domain_knowledge, tta=tta, fta=fta)

    def fit(self, loaders, transforms=None):
        """
        No training in zero-shot prediction
        """
        return

    def predict_image(self, loader):
        X, refs = self.extract_vision_features(loader)

        vis.vis_tsne(X, self.text_embeds.cpu().numpy(), refs)
        exit(0)

        mem_ratio_list=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
        beta_list=[3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5]
        for mem_ratio in mem_ratio_list:
            print('---',mem_ratio)
            self.obtain_prob_dmn(X, refs,mem_ratio=mem_ratio,beta=5.5)
        #self.obtain_prob_dmn(X, refs,mem_ratio=0.3,beta=5.5)
        #return self.obtain_label(X, refs, cluster=cluster)
    
    def obtain_prob_dmn(self, all_fea, all_label, mem_ratio=0.3, beta=5.5):
        '''
        all_fea: normalized
        '''
        all_fea = torch.tensor(all_fea).to(device)
        text_embeds = self.text_embeds.to(device)
        text_embeds_norm = torch.norm(text_embeds, p=2, dim=1, keepdim=True)
        text_embeds = text_embeds / text_embeds_norm
        
        with torch.no_grad():
            score = torch.matmul(all_fea, text_embeds.to(device).t()) * self.model.logit_scale.exp()###self.text_embeds
        all_output_text = torch.softmax(score, dim=-1)

        metrics_fold = evaluate(all_label, all_output_text.cpu().numpy(), 'classification')
        print(metrics_fold)

        #dmn
        K = all_output_text.size(1)
        _, pred_text = torch.max(all_output_text, 1)
        all_entropy = loss.Entropy(all_output_text)
        fea_mem = []
        for cls in range(K):
            cls_pred_num = (pred_text==cls).sum().item()
            cls_mem_num = int(cls_pred_num * mem_ratio)
            cls_entropy = all_entropy[pred_text==cls]
            cls_fea = all_fea[pred_text==cls]
            _, indice = torch.sort(cls_entropy)
            cls_fea = cls_fea[indice[:cls_mem_num]]
            fea_mem.append(cls_fea)
        fea_mem = stack_non_uniform(fea_mem)


        similarity_matrix = (all_fea.unsqueeze(1).unsqueeze(1) * fea_mem).sum(-1)
        similarity_matrix = torch.exp(-beta * (-similarity_matrix + 1))
        adaptive_image_feat = (fea_mem * similarity_matrix.unsqueeze(-1)).sum(-2)
        ## torch.Size([N, class, dim])
        adaptive_image_feat = adaptive_image_feat / adaptive_image_feat.norm(dim=-1, keepdim=True)
        
        ###adaptive_image_feat = adaptive_image_feat * text_embeds_norm
        with torch.no_grad():
            logits = torch.matmul(all_fea.unsqueeze(1), adaptive_image_feat.transpose(1,2)) * self.model.logit_scale.exp()
        #(N,1,K)

        all_output_adaptive = logits.squeeze().softmax(dim=-1)

        metrics_fold = evaluate(all_label, all_output_adaptive.cpu().numpy(), 'classification')
        print(metrics_fold)

        
        #initc = initc * text_embeds_norm
        """ for i in range(1,10):
            alpha=i*0.1
            print('---',alpha) """
        alpha=0.6
        metrics_fold = evaluate(all_label, (alpha*all_output_text+(1-alpha)*all_output_adaptive).cpu().numpy(), 'classification')
        print(metrics_fold)
        
        self.fea_mem = fea_mem

        return (all_output_text+all_output_adaptive)/2
    
    def obtain_label(self, all_fea, all_label, cluster=True):
        '''
        all_fea: normalized
        '''
        all_fea = torch.tensor(all_fea).to(device)
        text_embeds = self.text_embeds.to(device)
        text_embeds_norm = torch.norm(text_embeds, p=2, dim=1, keepdim=True)
        text_embeds = text_embeds / text_embeds_norm
        
        with torch.no_grad():
            score = torch.matmul(all_fea, self.text_embeds.to(device).t()) * self.model.logit_scale.exp()
        all_output = torch.softmax(score, dim=-1)

        if not cluster:
            return all_output
        
        all_output_text_embeds = all_output

        #_, predict = torch.max(all_output, 1)
        metrics_fold = evaluate(all_label, all_output.cpu().numpy(), 'classification')
        print(metrics_fold)

        all_fea = all_fea.cpu().numpy()
        K = all_output.size(1)
        aff = all_output.cpu().numpy()

        for _ in range(2):
            initc = aff.transpose().dot(all_fea)
            initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
            dd = cdist(all_fea, initc, 'cosine')
            predict = dd.argmin(axis=1)

            aff = np.eye(K)[predict]

        initc = torch.tensor(initc).to(device)
        initc = initc / torch.norm(initc, p=2, dim=1, keepdim=True)
        initc = initc * text_embeds_norm
        initc = initc.float()
        with torch.no_grad():
            score = torch.matmul(torch.tensor(all_fea).to(device), initc.t()) * self.model.logit_scale.exp()
        all_output = torch.softmax(score, dim=-1)

        metrics_fold = evaluate(all_label, ((all_output+all_output_text_embeds)/2).cpu().numpy(), 'classification')
        print(metrics_fold)
        

        self.text_embeds_cluster = initc

        return (all_output+all_output_text_embeds)/2

    def predict(self, loader, transforms=None, return_cam=False):

        if self.tta:
            scores = []
            for i in range(self.number_augmentations):
                X, refs = self.extract_vision_features(loader, transforms=augmentations_pretraining)
                X = torch.tensor(X).to(device)
                with torch.no_grad():
                    score_i = torch.matmul(torch.tensor(X), self.text_embeds.t()) * self.model.logit_scale.exp()
                scores.append(score_i.unsqueeze(-1))
            score = torch.concat(scores, -1).mean(-1)
        else:
            if return_cam:
                X, refs, features = self.extract_vision_features(loader, return_last_fea=True)
            else:
                X, refs = self.extract_vision_features(loader)

            #vis.vis_tsne(X, self.text_embeds.cpu().numpy(), refs)

            X = torch.tensor(X).to(device)
            with torch.no_grad():
                score = torch.matmul(X, self.text_embeds.t().to(device)) * self.model.logit_scale.exp()
                if return_cam:
                    patch_score = torch.einsum('bhwc,ck->bhwk', features, self.text_embeds.t().to(device)) * self.model.logit_scale.exp()

        # Softmax probs from scores
        preds = torch.softmax(score, dim=-1)
        preds = preds.detach().cpu().numpy()
        if return_cam:
            patch_preds = torch.softmax(patch_score, dim=-1)
            patch_preds = patch_preds.detach().cpu().numpy()
            return refs, preds, patch_preds
        return refs, preds

    def predict_batch(self, images, transforms=None, return_cam=False):

        if return_cam:
            X, proj_fea = self.extract_vision_features_batch(images, return_last_fea=True)
        else:
            X = self.extract_vision_features_batch(images)
        with torch.no_grad():

            score = torch.matmul(X, self.text_embeds.t().to(device)) * self.model.logit_scale.exp()
            #score_cluster = torch.matmul(X, self.text_embeds_cluster.t()) * self.model.logit_scale.exp()
            
            
            if return_cam:
                patch_score = torch.einsum('bhwc,ck->bhwk', proj_fea, self.text_embeds.t().to(device)) * self.model.logit_scale.exp()

        # Softmax probs from scores
        preds = torch.softmax(score, dim=-1)
        #preds_cluster = torch.softmax(score_cluster, dim=-1)
        #preds = 0.3*preds + 0.7*preds_cluster
        #preds = preds.detach().cpu().numpy()
        if return_cam:
            patch_preds = torch.softmax(patch_score, dim=-1)
            #patch_preds = patch_preds.detach().cpu().numpy()
            return preds, patch_preds
        return preds

class ClipAdapter(LanguageAdapterWrapper):
    def __init__(self, model, targets, domain_knowledge=False, tta=False, fta=False):
        super().__init__(model, targets, domain_knowledge=domain_knowledge, tta=tta, fta=fta)

        self.c_in = self.model.vision_model.out_dim
        self.reduction = 4
        self.ratio = 0.2

        # Set adapter
        self.adapter = torch.nn.Sequential(torch.nn.Linear(self.c_in, self.c_in // self.reduction, bias=False),
                                           torch.nn.ReLU(inplace=True),
                                           torch.nn.Linear(self.c_in // self.reduction, self.c_in, bias=False),
                                           torch.nn.ReLU(inplace=True)).to(device)

    def predict(self, loader, transforms=None):

        if self.tta:
            scores = []
            for i in range(self.number_augmentations):
                X, refs = self.extract_vision_features(loader, transforms=augmentations_pretraining)
                X = torch.tensor(X).to(device)
                with torch.no_grad():
                    # Compute residual CLIP-Adapter
                    X = self.residual_adapter(X)
                    # Compute similarity
                    score_i = torch.matmul(torch.tensor(X), self.text_embeds.t()) * self.model.logit_scale.exp()
                scores.append(score_i.unsqueeze(-1))
            score = torch.concat(scores, -1).mean(-1)
        else:
            X, refs = self.extract_vision_features(loader)
            X = torch.tensor(X).to(device)
            with torch.no_grad():
                # Compute residual CLIP-Adapter
                X = self.residual_adapter(X)
                # Compute similarity
                score = torch.matmul(torch.tensor(X), self.text_embeds.t()) * self.model.logit_scale.exp()

        # Softmax probs from scores
        preds = torch.softmax(score, dim=-1)
        preds = preds.detach().cpu().numpy()

        return refs, preds

    def train(self, X, Y):
        X = torch.tensor(X)
        Y = torch.tensor(Y)

        # TRAINING
        epochs, lr, bs = 40, 0.001, 1

        optimizer = torch.optim.AdamW(self.adapter.parameters(), lr=lr, eps=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs * X.shape[0])

        indexes = np.arange(0, X.shape[0])
        random.shuffle(indexes)
        for i_epoch in range(epochs):
            loss_epoch = 0.0
            for i_sample in range(X.shape[0]):
                X_batch = X[indexes[i_sample], :].unsqueeze(0).to(device)
                target = Y[indexes[i_sample]].unsqueeze(0).to(device)

                # Compute residual CLIP-Adapter
                X_batch = self.residual_adapter(X_batch)

                # Compute logits
                logits = self.model.logit_scale.exp() * X_batch @ self.text_embeds.t().to(device)

                # Compute loss
                loss = torch.nn.functional.cross_entropy(logits, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                loss_epoch += loss.item()/X.shape[0]

            print('loss=%2.5f' % loss_epoch, end="\n")

    def residual_adapter(self, X):
        # Compute residual CLIP-Adapter
        X_res = self.adapter(X)
        X = self.ratio * X_res + (1 - self.ratio) * X
        X = X / X.norm(dim=-1, keepdim=True)
        return X


class TipAdapter(LanguageAdapterWrapper):
    def __init__(self, model, targets, domain_knowledge=False, tta=False, fta=False, train=False):
        super().__init__(model, targets, domain_knowledge=domain_knowledge, tta=tta, fta=fta)

        self.beta = 5
        self.alpha = 1
        self.train_tip = train

        # Init cache values
        self.cache_keys = []
        self.cache_values = []
        self.adapter_layer = []

    def predict(self, loader, transforms=None):

        if self.tta:
            scores = []
            for i in range(self.number_augmentations):
                X, refs = self.extract_vision_features(loader, transforms=augmentations_pretraining)
                X = torch.tensor(X).to(device)
                with torch.no_grad():
                    score_i = self.adapter(X)
                scores.append(score_i.unsqueeze(-1))
            score = torch.concat(scores, -1).mean(-1)
        else:
            X, refs = self.extract_vision_features(loader, transforms=augmentations_pretraining)
            X = torch.tensor(X).to(device)
            with torch.no_grad():
                score = self.adapter(X)

        # Softmax probs from scores
        preds = torch.softmax(score, dim=-1)
        preds = preds.detach().cpu().numpy()

        return refs, preds

    def train(self, X, Y):
        X = torch.tensor(X)
        Y = torch.tensor(Y)

        self.cache_keys = torch.transpose(X, 1, 0).to(torch.float32).to(device)
        self.cache_values = torch.nn.functional.one_hot(Y).to(torch.float32).to(device)

        if self.train_tip:

            # TRAINING
            epochs, lr, bs = 40, 0.001, 1

            # Enable the cached keys to be learnable
            adapter_layer = torch.nn.Linear(self.cache_keys.shape[0], self.cache_keys.shape[1], bias=False).to(device)
            adapter_layer.weight = torch.nn.Parameter(self.cache_keys.t())
            adapter_layer = adapter_layer.to(device)

            optimizer = torch.optim.AdamW(adapter_layer.parameters(), lr=lr, eps=1e-4)

            indexes = np.arange(0, self.cache_keys.shape[1])
            random.shuffle(indexes)
            for i_epoch in range(epochs):
                loss_epoch = 0.0
                for i_sample in range(self.cache_keys.shape[1]):
                    image = self.cache_keys[:, indexes[i_sample]].unsqueeze(0).to(device)
                    target = self.cache_values[indexes[i_sample], :].argmax().unsqueeze(0).to(device)

                    # Zero-shot CLIP
                    clip_logits = self.model.logit_scale.exp() * (image @ self.text_embeds.t())

                    # Tip-Adapter
                    affinity = adapter_layer(image)
                    cache_logits = torch.exp(((-1) * (self.beta - self.beta * affinity))) @ self.cache_values
                    cache_logits /= X.shape[0]
                    cache_logits *= self.model.logit_scale.exp()

                    tip_logits = clip_logits + cache_logits * self.alpha

                    loss = torch.nn.functional.cross_entropy(tip_logits, target)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    loss_epoch += loss.item()/self.cache_keys.shape[1]

                print('loss=%2.5f' % loss_epoch, end="\n")

            # Storage trained adapter
            self.adapter_layer = adapter_layer

    def adapter(self, X):
        # Zero-shot CLIP
        clip_logits = 100 * (X @ self.text_embeds.t().to(device))

        # Tip-Adapter
        if not self.train_tip:
            affinity = X @ self.cache_keys
        else:
            affinity = self.adapter_layer(X)

        cache_logits = torch.exp(((-1) * (self.beta - self.beta * affinity))) @ self.cache_values
        logits = clip_logits + cache_logits * self.alpha

        return logits
    

###
def stack_non_uniform(tensors):
    # 找到最大长度
    max_length = max(tensor.size(0) for tensor in tensors)

    # 获取特征维度 D
    D = tensors[0].size(1)

    # 将每个张量补零到 (max_length, D)
    padded_tensors = []
    for tensor in tensors:
        # 创建一个全零张量
        padded_tensor = torch.zeros((max_length, D), dtype=tensor.dtype, device=tensor.device)
        # 将原始张量的数据复制到全零张量中
        padded_tensor[:tensor.size(0), :] = tensor
        padded_tensors.append(padded_tensor)

    # 堆叠张量，最终形状为 (k, max_length, D)
    result = torch.stack(padded_tensors)

    print("Result shape:", result.shape)

    return result

