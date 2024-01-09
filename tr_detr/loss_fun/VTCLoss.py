import torch
import torch.nn as nn
import torch.nn.functional as F

class VTCLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(VTCLoss, self).__init__()
        self.temperature = temperature

    def forward(self, src_txt, src_vid):
        # src_txt: (bs, h_dim)
        # src_vid: (bs, h_dim)
        bs = src_txt.size(0)
        h_dim = src_txt.size(1)
        # normalize the feature vectors
        src_txt = F.normalize(src_txt, dim=1)
        src_vid = F.normalize(src_vid, dim=1)
        # compute the similarity matrix
        sim_mat = torch.mm(src_txt, src_vid.t()) # (bs, bs)
        # create the positive and negative masks
        pos_mask = torch.eye(bs).bool().to(sim_mat.device) # (bs, bs)
        neg_mask = ~pos_mask # (bs, bs)
        # compute the logits and labels
        logits = sim_mat / self.temperature # (bs, bs)
        labels = torch.arange(bs).to(sim_mat.device) # (bs,)
        # compute the cross entropy loss for text-to-video and video-to-text
        loss_t2v = F.cross_entropy(logits, labels) # scalar
        loss_v2t = F.cross_entropy(logits.t(), labels) # scalar
        # return the average loss
        return (loss_t2v + loss_v2t) / 2
