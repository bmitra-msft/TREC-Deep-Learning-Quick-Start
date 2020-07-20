import torch
import torch.nn as nn


class SmoothRank(nn.Module):

    def __init__(self):
        super(SmoothRank, self).__init__()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, scores):
        x_0 = scores.unsqueeze(dim=-1)                                              # [Q x D] --> [Q x D x 1]
        x_1 = scores.unsqueeze(dim=-2)                                              # [Q x D] --> [Q x 1 x D]
        diff = x_1 - x_0                                                            # [Q x D x 1], [Q x 1 x D] --> [Q x D x D]
        is_lower = self.sigmoid(diff)                                               # [Q x D x D] --> [Q x D x D]
        ranks = torch.sum(is_lower, dim=-1) + 0.5                                   # [Q x D x D] --> [Q x D]
        return ranks


class SmoothMRRLoss(nn.Module):

    def __init__(self):
        super(SmoothMRRLoss, self).__init__()
        self.soft_ranker = SmoothRank()
        self.zero = nn.Parameter(torch.tensor([0], dtype=torch.float32), requires_grad=False)
        self.one = nn.Parameter(torch.tensor([1], dtype=torch.float32), requires_grad=False)

    def forward(self, scores, labels, agg=True):
        ranks = self.soft_ranker(scores)                                            # [Q x D] --> [Q x D]
        labels = torch.where(labels > 0, self.one, self.zero)                       # [Q x D] --> [Q x D]
        rr = labels / ranks                                                         # [Q x D], [Q x D] --> [Q x D]
        rr_max, _ = rr.max(dim=-1)                                                  # [Q x D] --> [Q]
        loss = 1 - rr_max                                                           # [Q] --> [Q]
        if agg:
            loss = loss.mean()                                                      # [Q] --> [1]
        return loss


class RankNetLoss(nn.Module):

    def __init__(self):
        super(RankNetLoss, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.zero = nn.Parameter(torch.tensor([0], dtype=torch.float32), requires_grad=False)
        self.one = nn.Parameter(torch.tensor([1], dtype=torch.float32), requires_grad=False)

    def forward(self, scores, labels, agg=True):
        x_0 = scores.unsqueeze(dim=-1)                                              # [Q x D] --> [Q x D x 1]
        x_1 = scores.unsqueeze(dim=-2)                                              # [Q x D] --> [Q x 1 x D]
        x = x_0 - x_1                                                               # [Q x D x 1], [Q x 1 x D] --> [Q x D x D]
        x = self.sigmoid(x)                                                         # [Q x D x D] --> [Q x D x D]
        x = -torch.log(x + 1e-6)                                                    # [Q x D x D] --> [Q x D x D]
        y_0 = labels.unsqueeze(dim=-1)                                              # [Q x D] --> [Q x D x 1]
        y_1 = labels.unsqueeze(dim=-2)                                              # [Q x D] --> [Q x 1 x D]
        y = y_0 - y_1                                                               # [Q x D x 1], [Q x 1 x D] --> [Q x D x D]
        pair_mask = torch.where(y > 0, self.one, self.zero)                         # [Q x D x D] --> [Q x D x D]
        num_pairs = pair_mask.sum(dim=-1)                                           # [Q x D x D] --> [Q x D]
        num_pairs = num_pairs.sum(dim=-1)                                           # [Q x D] --> [Q]
        num_pairs = torch.where(num_pairs > 0, num_pairs, self.one)                 # [Q] --> [Q]
        loss = x * pair_mask                                                        # [Q x D x D], [Q x D x D] --> [Q x D x D]
        loss = loss.sum(dim=-1)                                                     # [Q x D x D] --> [Q x D]
        loss = loss.sum(dim=-1)                                                     # [Q x D] --> [Q]
        loss = loss / num_pairs                                                     # [Q], [Q] --> [Q]
        if agg:
            loss = loss.mean()                                                      # [Q] --> [1]
        return loss


class MarginLoss(nn.Module):

    def __init__(self):
        super(MarginLoss, self).__init__()
        self.zero = nn.Parameter(torch.tensor([0], dtype=torch.float32), requires_grad=False)
        self.one = nn.Parameter(torch.tensor([1], dtype=torch.float32), requires_grad=False)
        self.neg_one = nn.Parameter(torch.tensor([-1], dtype=torch.float32), requires_grad=False)
        self.margin = nn.Parameter(torch.tensor([0.1], dtype=torch.float32), requires_grad=False)

    def forward(self, scores, labels, agg=True):
        x_0 = scores.unsqueeze(dim=-1)                                              # [Q x D] --> [Q x D x 1]
        x_1 = scores.unsqueeze(dim=-2)                                              # [Q x D] --> [Q x 1 x D]
        x = x_0 - x_1                                                               # [Q x D x 1], [Q x 1 x D] --> [Q x D x D]
        y_0 = labels.unsqueeze(dim=-1)                                              # [Q x D] --> [Q x D x 1]
        y_1 = labels.unsqueeze(dim=-2)                                              # [Q x D] --> [Q x 1 x D]
        y = y_0 - y_1                                                               # [Q x D x 1], [Q x 1 x D] --> [Q x D x D]
        y = torch.where(y > 0, self.one, y)                                         # [Q x D x D] --> [Q x D x D]
        y = torch.where(y < 0, self.neg_one, y)                                     # [Q x D x D] --> [Q x D x D]
        loss = y * x                                                                # [Q x D x D], [Q x D x D] --> [Q x D x D]
        loss = self.margin - loss                                                   # [1], [Q x D x D] --> [Q x D x D]
        loss = torch.where(loss < 0, self.zero, loss)                               # [Q x D x D] --> [Q x D x D]
        loss = loss.sum(dim=-1)                                                     # [Q x D x D] --> [Q x D]
        loss = loss.sum(dim=-1)                                                     # [Q x D] --> [Q]
        num_pairs = torch.where(y < 0, self.one, y)                                 # [Q x D x D] --> [Q x D x D]
        num_pairs = num_pairs.sum(dim=-1)                                           # [Q x D x D] --> [Q x D]
        num_pairs = num_pairs.sum(dim=-1)                                           # [Q x D] --> [Q]
        loss = loss / num_pairs                                                     # [Q], [Q] --> [Q]
        if agg:
            loss = loss.mean()                                                      # [Q] --> [1]
        return loss
