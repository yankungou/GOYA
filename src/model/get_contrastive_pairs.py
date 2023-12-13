# code is adopted from https://github.com/KinWaiCheuk/pytorch-triplet-loss/blob/master/TNN/Mining.py
import torch


def get_soft_label_equal_mtx(labels, dist_mtx, ther):
    """Return a 2D mask where dist_mtx(labels[i], labels[j]) <= ther.
    (i, j) is the same label if
        - dist_mtx(labels[i], labels[j]) <= ther
    Args:
        labels: `Tensor` with shape [batch_size]
        dist_mtx: `Tensor` with shape [all_title_num, all_title_num]
    """
    labels_cpu = labels.cpu()
    sub_dist_mtx = dist_mtx[labels_cpu[:, None], labels_cpu]
    label_equal = sub_dist_mtx <= ther
    label_equal = label_equal.to(labels.device)
    
    return label_equal


def get_pair_mask(labels, label_equal=None):
    """Return a 2D mask where mask[a, p] is True if a and p are positive.
    A pair (i, j) is valid if:
        - i, j are distinct
        - labels[i] == labels[j]
    Args:
        labels: `Tensor` with shape [batch_size]
    """
    # Check that i, j are distinct
    indices_equal = torch.eye(labels.size(0), device=labels.device).bool()
    indices_not_equal = ~indices_equal

    if label_equal is None:
        label_equal = labels.unsqueeze(0) == labels.unsqueeze(1)

    return label_equal & indices_not_equal


def get_indices_tuples(mask):
    """
    Args:
        mask: 2D mask where mask[a, p] is True if a and p are positive.
    """
    a1, p = torch.where(mask.triu() == True)

    diag_false = ~torch.eye(mask.size(0), device=mask.device).bool()
    neg_mask = ~mask & diag_false
    a2, n = torch.where(neg_mask.triu() == True)
    
    # print('positive num', p.shape)
    # print('negative num', n.shape)

    return (a1, p, a2, n)   


def get_cs_sep_indices_tuples(content_labels, style_labels, dist_mtx, ther=0.25):
    '''style + soft content'''
    content_label_equal = get_soft_label_equal_mtx(content_labels, dist_mtx, ther)
    content_pair_mask = get_pair_mask(content_labels, label_equal=content_label_equal)
    style_pair_mask = get_pair_mask(style_labels)
    content_indices_tuple = get_indices_tuples(content_pair_mask)
    style_indices_tuple = get_indices_tuples(style_pair_mask)

    return content_indices_tuple, style_indices_tuple
