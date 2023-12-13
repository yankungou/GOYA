# code is adopted from https://github.com/KinWaiCheuk/pytorch-triplet-loss/blob/master/TNN/Mining.py
import torch
import torch.nn.functional as F


def cosine_pairwise(x):
    x = x.float().unsqueeze(0)
    x = x.permute((1, 2, 0))
    cos_sim_pairwise = torch.nn.functional.cosine_similarity(x, x.unsqueeze(1), dim=-2)
    cos_sim_pairwise = cos_sim_pairwise.permute((2, 0, 1))[0]
    return 1 - cos_sim_pairwise


def _pairwise_distances(embeddings):
    """Compute the 2D matrix of distances between all the embeddings.
    Args:
        embeddings: tensor of shape (batch_size, embed_dim)
    Returns:
        pairwise_distances: tensor of shape (batch_size, batch_size)
    """
    distances = cosine_pairwise(embeddings)

    return distances


def _get_triplet_mask(labels):
    """Return a 3D mask where mask[a, p, n] is True iff the triplet (a, p, n) is valid.
    A triplet (i, j, k) is valid if:
        - i, j, k are distinct
        - labels[i] == labels[j] and labels[i] != labels[k]
    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]
    """
    # Check that i, j and k are distinct
    indices_equal = torch.eye(labels.size(0), device=labels.device).bool()
    indices_not_equal = ~indices_equal
    i_not_equal_j = indices_not_equal.unsqueeze(2)
    i_not_equal_k = indices_not_equal.unsqueeze(1)
    j_not_equal_k = indices_not_equal.unsqueeze(0)

    distinct_indices = (i_not_equal_j & i_not_equal_k) & j_not_equal_k

    label_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
    i_equal_j = label_equal.unsqueeze(2)
    i_equal_k = label_equal.unsqueeze(1)

    valid_labels = ~i_equal_k & i_equal_j

    return valid_labels & distinct_indices


def batch_all_cs_triplet_loss(mask, embeddings, margin):
    """Build the triplet loss over a batch of embeddings.
    We generate all the valid triplets and average the loss over the positive ones.
    Args:
        labels: labels of the batch, of size (batch_size,)
        embeddings: tensor of shape (batch_size, embed_dim)
        margin: margin for triplet loss
    Returns:
        triplet_loss: scalar tensor containing the triplet loss
    """
    # Get the pairwise distance matrix
    pairwise_dist = _pairwise_distances(embeddings)

    anchor_positive_dist = pairwise_dist.unsqueeze(2)
    anchor_negative_dist = pairwise_dist.unsqueeze(1)

    # Compute a 3D tensor of size (batch_size, batch_size, batch_size)
    # triplet_loss[i, j, k] will contain the triplet loss of anchor=i, positive=j, negative=k
    # Uses broadcasting where the 1st argument has shape (batch_size, batch_size, 1)
    # and the 2nd (batch_size, 1, batch_size)
    triplet_loss = anchor_positive_dist - anchor_negative_dist + margin

    # Put to zero the invalid triplets
    # (where label(a) != label(p) or label(n) == label(a) or a == p)
    triplet_loss = mask.float() * triplet_loss

    # Remove negative losses (i.e. the easy triplets)
    triplet_loss = F.relu(triplet_loss)

    # Count number of positive triplets (where triplet_loss > 0)
    valid_triplets = triplet_loss[triplet_loss > 1e-16]
    num_positive_triplets = valid_triplets.size(0)
    num_valid_triplets = mask.sum()

    fraction_positive_triplets = num_positive_triplets / (num_valid_triplets.float() + 1e-16)

    # Get final mean triplet loss over the positive valid triplets
    triplet_loss = triplet_loss.sum() / (num_valid_triplets + 1e-16)

    return triplet_loss


# ------soft content------- #
def _get_soft_label_equal_mtx(labels, dist_mtx, ther):
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


def _get_soft_triplet_mask(labels, dist_mtx, ther):
    """Return a 3D mask where mask[a, p, n] is True iff the triplet (a, p, n) is valid.
    A triplet (i, j, k) is valid if:
        - i, j, k are distinct
        - labels[i] == labels[j] and labels[i] != labels[k]
    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]
    """
    # Check that i, j and k are distinct
    indices_equal = torch.eye(labels.size(0), device=labels.device).bool()
    indices_not_equal = ~indices_equal
    i_not_equal_j = indices_not_equal.unsqueeze(2)
    i_not_equal_k = indices_not_equal.unsqueeze(1)
    j_not_equal_k = indices_not_equal.unsqueeze(0)

    distinct_indices = (i_not_equal_j & i_not_equal_k) & j_not_equal_k

    label_equal = _get_soft_label_equal_mtx(labels, dist_mtx, ther)
    i_equal_j = label_equal.unsqueeze(2)
    i_equal_k = label_equal.unsqueeze(1)

    valid_labels = ~i_equal_k & i_equal_j

    return valid_labels & distinct_indices


def get_soft_cs_triplet_mask(content_labels, style_labels, dist_mtx, ther=0.25):
    '''get C, A, S mask, content (title) triplets are soft triplets
    A: content_a, style_a
    C: content_p, style_n
    S: content_n, style_p
    '''
    # triplet_loss[i, j, k] will contain the triplet loss of anchor=i, positive=j, negative=k
    # (where label(a) != label(p) or label(n) == label(a) or a == p)
    content_mask = _get_soft_triplet_mask(content_labels, dist_mtx, ther)
    style_mask = _get_triplet_mask(style_labels)
    style_mask = torch.transpose(style_mask, 1, 2)
    valid_mask = content_mask & style_mask
    
    valid_content_mask = valid_mask
    valid_style_mask = torch.transpose(valid_mask, 1, 2)

    return valid_content_mask, valid_style_mask
