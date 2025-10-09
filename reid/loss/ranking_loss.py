# -*- coding: utf-8 -*-
"""
Created on Sat., Aug. 17(rd), 2019 at 15:41

@author: zifyloo
"""

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class RankingLoss(nn.Module):

    def __init__(self, margin):
        super(RankingLoss, self).__init__()
        self.margin = margin

    def semi_hard_negative(self, loss):
        negative_index = np.where(np.logical_and(loss < self.margin, loss > 0))[0]
        return np.random.choice(negative_index) if len(negative_index) > 0 else None

    def get_triplets(self, similarity, labels):
        similarity = similarity.cpu().data.numpy()

        labels = labels.cpu().data.numpy()
        triplets = []

        for idx, label in enumerate(labels):  # same class calculate together

            negative = np.where(labels != label)[0]

            ap_sim = similarity[idx, idx]
            # print(ap_combination_list.shape, ap_distances_list.shape)

            loss = similarity[idx, negative] - ap_sim + self.margin

            negetive_index = self.semi_hard_negative(loss)

            if negetive_index is not None:
                triplets.append([idx, idx, negative[negetive_index]])

        if len(triplets) == 0:
            triplets.append([idx, idx, negative[0]])

        triplets = np.array(triplets)

        return torch.LongTensor(triplets)

    def forward(self, image_features, text_features, logit_scale, label):
        image_norm = image_features / (image_features.norm(dim=1, keepdim=True) + 1e-8)
        text_norm = text_features / (text_features.norm(dim=1, keepdim=True) + 1e-8)

        similarity = logit_scale * image_norm @ text_norm.t()

        image_triplets = self.get_triplets(similarity, label)
        text_triplets = self.get_triplets(similarity.t(), label)

        # print(image_triplets.size(), text_triplets.size())
        image_anchor_loss = F.relu(self.margin
                            - similarity[image_triplets[:, 0], image_triplets[:, 1]]
                            + similarity[image_triplets[:, 0], image_triplets[:, 2]])

        texy_anchor_loss = F.relu(self.margin
                            - similarity[text_triplets[:, 0], text_triplets[:, 1]]
                            + similarity[text_triplets[:, 0], text_triplets[:, 2]])

        loss = torch.sum(image_anchor_loss) + torch.sum(texy_anchor_loss)
        # loss = CMPM_loss + CMPC_loss

        return loss.item()


"""
# test code
SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)

image_embeddings = torch.rand(8, 512)
text_embeddings = torch.rand(8, 512)
label = torch.LongTensor([1, 2, 2, 2, 2, 2, 2, 2])

triplet_loss = RankingLoss(0.3)
print(triplet_loss(image_embeddings, text_embeddings, label))
"""

