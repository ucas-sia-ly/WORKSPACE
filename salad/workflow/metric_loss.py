"""The repository's MultiSimilarityLoss + MultiSimilarityMiner, in native PyTorch."""

import torch
import torch.nn.functional as F


def multi_similarity_loss(descriptors, labels, alpha=1.0, beta=50.0, base=0.0, epsilon=0.1):
    """Average all anchor losses after cosine hard-pair mining.

    Defaults match utils/losses.py. Computation uses float32 under mixed precision.
    A batch without useful pairs returns a differentiable zero.
    """
    with torch.autocast(device_type=descriptors.device.type, enabled=False):
        return _multi_similarity_loss(descriptors.float(), labels, alpha, beta, base, epsilon)


def _multi_similarity_loss(descriptors, labels, alpha, beta, base, epsilon):
    descriptors = F.normalize(descriptors, dim=1)
    similarity = descriptors @ descriptors.T
    same = labels[:, None] == labels[None, :]
    positive = same & ~torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    negative = ~same
    with torch.no_grad():
        hardest_negative = similarity.masked_fill(~negative, -torch.inf).max(dim=1).values
        hardest_positive = similarity.masked_fill(~positive, torch.inf).min(dim=1).values
        positive = positive & (similarity < hardest_negative[:, None] + epsilon)
        negative = negative & (similarity > hardest_positive[:, None] - epsilon)
    zero = similarity.new_zeros((len(labels), 1))
    positive_terms = (-alpha * (similarity - base)).masked_fill(~positive, -torch.inf)
    negative_terms = (beta * (similarity - base)).masked_fill(~negative, -torch.inf)
    loss = torch.logsumexp(torch.cat((zero, positive_terms), dim=1), dim=1) / alpha
    loss = loss + torch.logsumexp(torch.cat((zero, negative_terms), dim=1), dim=1) / beta
    return loss.mean()
