import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

from protonets.models import register_model
from . import wasserstein

from .utils import euclidean_dist

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)

class Protonet(nn.Module):
    def __init__(self, encoder):
        super(Protonet, self).__init__()

        self.encoder = encoder

    def embed(self, sample, raw_input=False):
        '''
        Compute embeddings of sample z = h(x)

        :param sample: dictionary
            sample['xs'] of size (n_class,  n_support,     channel, height, width),
            sample['xq'] of size (n_class,  n_query,       channel, height, width)
        :param raw_input: if True, then do nothing, simple flatten x
        :return: dictionary
            sample['zs'] of size (n_class,  n_support,  latent_dims),
            sample['zq'] of size (n_class,  n_query,  latent_dims),
        '''

        # Cuda ?

        xs = sample['xs'] # support
        xq = sample['xq'] # query

        n_class = xs.size(0)
        assert xq.size(0) == n_class
        n_support = xs.size(1)
        n_query = xq.size(1)

        x = torch.cat([xs.view(n_class * n_support, *xs.size()[2:]),
                       xq.view(n_class * n_query, *xq.size()[2:])], 0)

        if raw_input:
            # No embedding, z = x
            z = x.view(len(x), -1)
        else:
            z = self.encoder.forward(x)

        zs = z[:n_class * n_support].view(n_class, n_support, -1)
        zq = z[n_class * n_support:].view(n_class, n_query, -1)

        embedded_sample = {
            'zs': zs,
            'zq': zq,
            'class': sample['class']
        }

        return embedded_sample


    def supervised_loss(self, embedded_sample, regularization, supervised_sinkhorn_loss):

        # Prepare data
        z_support = embedded_sample['zs'] # support
        z_query = embedded_sample['zq'] # query

        n_class = z_support.size(0)
        assert z_query.size(0) == n_class, 'need same number of classes'
        n_support = z_support.size(1)
        n_query = z_query.size(1)
        z_dim = z_support.size(-1)

        target_inds_support = torch.arange(0, n_class).view(n_class, 1, 1).expand(n_class, n_support, 1).long().to(z_support.device)
        target_inds_query = torch.arange(0, n_class).view(n_class, 1, 1).expand(n_class, n_query, 1).long().to(z_support.device)

        # Compute prototypes
        z_proto = z_support.view(n_class, n_support, z_dim).mean(1)
        # THis was clearly wrong!
        #class_variance = ((z_proto - z_support.view(n_class, n_support, z_dim).mean(1)[:, None, :])**2).mean()
        class_variance = ((z_support - z_proto[:, None, :])**2).mean()

        # Compute query-prototype distances
        query_dists = euclidean_dist(z_query, z_proto)

        # Assign query points to prototypes
        if not supervised_sinkhorn_loss:
            # Default protonet. Assignment is softmax on squared cluster-sample distance
            # divide/multiply by correct temperature/regularization
            log_p_y_query = F.log_softmax(-query_dists*regularization, dim=1).view(n_class, n_query, -1)
        else:
            # This part changes
            # Assignment is now regularized, optimal transport, but transportation cost is given by distance matrix.
            # this differs from previously because the regularization is slightly different
            # todo: grid search on iterations
            __, __, log_assignment, __, __ = wasserstein.compute_sinkhorn_stable(
                query_dists, regularization=regularization, iterations=10)
            log_p_y_query = log_assignment.view(n_class, n_query, -1)

        # Supervised Loss (Query/Validation)
        supervised_loss = -log_p_y_query.gather(2, target_inds_query).squeeze().view(-1).mean()

        # Supervised Accuracy and Predictions
        _, y_hat_query = log_p_y_query.max(2)
        supervised_accuracy = torch.eq(y_hat_query, target_inds_query.squeeze()).float().mean()

        return supervised_loss, {
            'SupervisedLoss': supervised_loss.item(),
            'SupervisedAcc': supervised_accuracy.item(),
            'ClassVariance': class_variance
        }



@register_model('protonet_conv')
def load_protonet_conv(**kwargs):
    x_dim = kwargs['x_dim']
    hid_dim = kwargs['hid_dim']
    z_dim = kwargs['z_dim']

    def conv_block(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

    encoder = nn.Sequential(
        conv_block(x_dim[0], hid_dim),
        conv_block(hid_dim, hid_dim),
        conv_block(hid_dim, hid_dim),
        conv_block(hid_dim, z_dim),
        Flatten()
    )

    return Protonet(encoder)


class ClusterNet(Protonet):
    def __init__(self, encoder):
        super(ClusterNet, self).__init__(encoder)

    def clustering_loss(self, embedded_sample, regularization, supervised_sinkhorn_loss):
        '''
        This function returns results for two settings (simultaneously):
        - Learning to Cluster:
            - cluster support set.
            - p(y=cluster k | x) given either by Sinkhorn or Softmax
            - reveal support set labels (for evaluation)
            - find optimal matching between predicted clusters and support set labels
            -> Score is clustering accuracy on support set
        - Unsupervised Few-Shot Learning: cluster support set
            - cluster support set, get centroids.
            - p(y=cluster k | x) given either by Sinkhorn or Softmax
            - reveal support set labels (for evaluation)
            - permute clusters accordingly
            - now classify query set data using centroids as prototypes.
            -> Score is supervised accuracy of query set.

        :param embedded_sample:
        :param regularization:
        :param supervised_sinkhorn_loss:
        :param raw_input:
        :return:
        '''

        # Prepare data
        z_support = embedded_sample['zs'] # support
        z_query = embedded_sample['zq'] # query

        n_class = z_support.size(0)
        assert z_query.size(0) == n_class, 'need same number of classes'
        n_support = z_support.size(1)
        n_query = z_query.size(1)
        z_dim = z_support.size(-1)

        target_inds_support = torch.arange(0, n_class).view(n_class, 1, 1).expand(n_class, n_support, 1).long().to(z_support.device)
        target_inds_query = torch.arange(0, n_class).view(n_class, 1, 1).expand(n_class, n_query, 1).long().to(z_support.device)

        # Build support set targets
        target_inds_dummy_support = np.zeros((n_class*n_query, n_class))
        target_inds_dummy_support[range(n_class*n_query), target_inds_support.cpu().numpy().flatten()] = 1. # transform targets to one-hot
        target_inds_dummy_support = torch.FloatTensor(target_inds_dummy_support).to(xs.device)

        # Cluster support set into clusters (both for learning to cluster and unsupervised few shot learning)
        z_centroid, data_centroid_assignment = wasserstein.cluster_wasserstein(z_support, n_class, stop_gradient=False, regularization=regularization)

        # Pairwise distance from query set to centroids
        support_dists = euclidean_dist(z_support, z_centroid)
        query_dists = euclidean_dist(z_query, z_centroid)

        # Assign support set points to centroids (using either Sinkhorn or Softmax)
        if supervised_sinkhorn_loss:

            # Optimal assignment (could have kept previous result probably)
            __, __, log_assignment, __, __ = wasserstein.compute_sinkhorn_stable(
                support_dists, regularization=regularization, iterations=10)
            # Predictions are already the optimal assignment
            log_p_y_support = log_assignment.view(n_class, n_query, -1)

        else: # Classic softmax

            # Unpermuted Log Probabilities
            log_p_y_support = F.log_softmax(-support_dists*regularization, dim=1).view(n_class, n_query, -1)

        # Build accuracy permutation matrix (to match support with ground truth)
        __, y_hat_support = log_p_y_support.view(n_class * n_support, n_class).max(1)
        y_hat_support = y_hat_support.cpu().numpy()  # to numpy, no need to backprop anyways
        one_hot_prediction = torch.zeros((n_class * n_support, n_class)).to(z_support.device)
        one_hot_prediction[range(n_class * n_support), y_hat_support] = 1.
        accuracy_permutation_cost_support = -one_hot_prediction.view(n_class, n_support, n_class, 1) * target_inds_dummy_support.view(
            n_class, n_support, 1, n_class)
        accuracy_permutation_cost_support = accuracy_permutation_cost_support.sum(1).sum(0)

        # Use Hungarian algorithm to find best match
        __, __, cols_support = wasserstein.compute_hungarian(accuracy_permutation_cost_support)
        support_permuted_prediction = cols_support[y_hat_support]
        support_clustering_accuracy = (support_permuted_prediction == target_inds_support.cpu().numpy().flatten()).mean()

        # Now, run standard prototypical networks
        log_p_y_query = F.log_softmax(-query_dists*regularization, dim=1).view(n_class, n_query, -1)
        _, y_hat_query = log_p_y_query.max(2)
        y_hat_query = y_hat_query.cpu().numpy()

        # Permute predictions
        query_permuted_predictions = cols_support[y_hat_query]
        query_clustering_accuracy = (query_permuted_predictions == target_inds_query.cpu().numpy().flatten()).mean()

        # This was only useful when backpropping end-to-end.
        # # Build permutation cost matrix
        # permutation_cost = -log_p_y.view(n_class, n_query, n_class, 1) * target_inds_dummy.view(n_class, n_query, 1, n_class)
        # permutation_cost = permutation_cost.sum(1).sum(0)
        #
        # # Permuted log probabilities
        # loss_val_unnormalized, assignment, __, __, __ = wasserstein.compute_sinkhorn_stable(
        #     permutation_cost, regularization=100., iterations=10)
        #
        # loss_val = loss_val_unnormalized / n_query  # normalize so it looks like a normal cross entropy
        #
        # log_p_y_permuted = n_class * torch.matmul(log_p_y, assignment)
        #
        # # _, y_hat = log_p_y.max(2)
        # __, y_hat = log_p_y_permuted.max(2)
        #
        # acc_val = torch.eq(y_hat, target_inds.squeeze()).float().mean()

        # Compute best label assignment and corresponding loss
        # assignment[a,b] = 1_{a=\sigma(b)}
        # min_{\gamma} \sum_{a,b} permutation_cost[a,b] * assignment[a,b]
        # might not backprop through the graph though ...
        # it might or might not be equivalent due to the contraints (is it a critical point?)

        return None, {
            'SupportClusteringAcc': support_clustering_accuracy,
            'QueryClusteringAcc': query_clustering_accuracy
        }


@register_model('clusternet_conv')
def load_clusternet_conv(**kwargs):
    x_dim = kwargs['x_dim']
    hid_dim = kwargs['hid_dim']
    z_dim = kwargs['z_dim']

    def conv_block(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

    encoder = nn.Sequential(
        conv_block(x_dim[0], hid_dim),
        conv_block(hid_dim, hid_dim),
        conv_block(hid_dim, hid_dim),
        conv_block(hid_dim, z_dim),
        Flatten()
    )

    return ClusterNet(encoder)
