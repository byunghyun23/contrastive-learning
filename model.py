import torch
import numpy as np
from torch import nn


class RepresentativeModel(nn.Module):
    def __init__(self, embedding_dim):
        super(RepresentativeModel, self).__init__()
        self.embedding_dim = embedding_dim
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.projection_head = nn.Sequential(
            nn.Linear(256 * 8 * 8, embedding_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.projection_head(x)
        return x


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.5, batch_size=128):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.batch_size = batch_size
        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)

        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, zi, zj):
        N = 2 * self.batch_size

        z = torch.cat((zi, zj), dim=0)

        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)

        labels = torch.from_numpy(np.array([0] * N)).reshape(-1).to(positive_samples.device).long()

        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        return loss


class FineTunedModel(nn.Module):
    def __init__(self, embedding_dim, num_classes, representative_model):
        super(FineTunedModel, self).__init__()
        self.backbone = representative_model
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.Linear(embedding_dim // 2, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x