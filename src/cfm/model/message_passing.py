import math

import torch
import torch.nn.functional as F


def _segment_softmax(scores, indices, count):
    maxes = torch.full(
        (count, scores.shape[1]), -torch.inf, dtype=scores.dtype, device=scores.device
    )
    expanded = indices.unsqueeze(-1).expand_as(scores)
    maxes.scatter_reduce_(0, expanded, scores, reduce="amax", include_self=True)
    weights = (scores - maxes[indices]).exp()
    totals = torch.zeros_like(maxes).scatter_add_(0, expanded, weights)
    totals = totals.clamp_min(torch.finfo(weights.dtype).tiny)
    return weights / totals[indices]


def _segment_sum(values, indices, count):
    result = values.new_zeros((count, values.shape[1]))
    return result.index_add_(0, indices, values)


def aggregate_attention(context, indices, count, q_layer, k_layer, v_layer, out_layer,
                        dim, head, dropout=0.0, training=False):
    """Aggregate edge attention into worker or task representations."""
    q, k, v = [module(context).view(-1, head, dim) for module in (q_layer, k_layer, v_layer)]
    scores = (q * k).sum(-1) / math.sqrt(dim)
    weights = _segment_softmax(scores, indices, count)
    message = out_layer((weights.unsqueeze(-1) * v).reshape(-1, head * dim))
    message = F.dropout(message, p=dropout, training=training)
    return _segment_sum(message, indices, count)
