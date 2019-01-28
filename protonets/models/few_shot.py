import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

from protonets.models import register_model
from . import wasserstein
from protonets.models.vgg import VGGS

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


    def supervised_loss(self, embedded_sample, regularization):

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

        # We average variance over all n_class*n_support points, but not over z_dim (not necessarily meaningful for z_dim)
        # TODO: check if that's the best normalization - Now averaging everything except over dimensions
        #class_variance = ((z_support - z_proto[:, None, :])**2).mean(1).mean(0).sum()
        class_variance = ((z_support - z_proto[:, None, :])**2).mean()
        #old_class_variance = ((z_support - z_proto[:, None, :])**2).mean(2).mean(1).sum()

        # Compute support query-prototype distances
        support_dists = euclidean_dist(z_support.view(n_class*n_support, z_dim), z_proto)
        query_dists = euclidean_dist(z_query.view(n_class*n_query, z_dim), z_proto)

        # Assign query points to prototypes

        ############ Softmax conditionals ###################
        # Default protonet. Assignment is softmax on squared cluster-sample distance
        # divide/multiply by correct temperature/regularization
        softmax_log_p_y_query = F.log_softmax(-query_dists*regularization, dim=1).view(n_class, n_query, -1)

        # Supervised Loss (Query/Validation)
        softmax_supervised_loss = -softmax_log_p_y_query.gather(2, target_inds_query).squeeze().view(-1).mean()

        # Supervised Accuracy and Predictions
        _, softmax_y_hat_query = softmax_log_p_y_query.max(2)
        softmax_supervised_accuracy = torch.eq(softmax_y_hat_query, target_inds_query.squeeze()).float().mean()

        ############ Sinkhorn conditionals ##################
        # Assignment is now regularized with optimal transport, but transportation cost is given by distance matrix.
        # this differs from previously because the regularization is slightly different
        # todo: grid search on iterations
        __, __, log_assignment, __, __ = wasserstein.compute_sinkhorn_stable(
            query_dists, regularization=regularization, iterations=10)
        sinkhorn_log_p_y_query = log_assignment.view(n_class, n_query, -1)

        # Supervised Loss (Query/Validation)
        sinkhorn_supervised_loss = -sinkhorn_log_p_y_query.gather(2, target_inds_query).squeeze().view(-1).mean()

        # Supervised Accuracy and Predictions
        _, sinkhorn_y_hat_query = sinkhorn_log_p_y_query.max(2)
        sinkhorn_supervised_accuracy = torch.eq(sinkhorn_y_hat_query, target_inds_query.squeeze()).float().mean()

        ############ Two-step conditionals ##################
        # this is a potentially good surrogate loss for unsupervisewd few-shot learning setting
        # Use optimal assignment on support set to recompute new prototypes
        __, P, log_P, __, __ = wasserstein.compute_sinkhorn_stable(
            support_dists, regularization=regularization, iterations=10)
        support_to_prototype_assignments = P / P.sum(0, keepdim=True)

        two_step_prototypes = torch.mm(support_to_prototype_assignments.t(), z_support.view((n_class*n_support, -1)))

        two_step_query_dists = euclidean_dist(z_query.view(n_class * n_query, z_dim), two_step_prototypes)

        two_step_softmax_log_p_y_query = F.log_softmax(-two_step_query_dists*regularization, dim=1).view(n_class, n_query, -1)

        # Supervised Loss (Query/Validation)
        two_step_softmax_supervised_loss = -two_step_softmax_log_p_y_query.gather(2, target_inds_query).squeeze().view(-1).mean()

        # Supervised Accuracy and Predictions
        _, two_step_softmax_y_hat_query = two_step_softmax_log_p_y_query.max(2)
        two_step_softmax_supervised_accuracy = torch.eq(two_step_softmax_y_hat_query, target_inds_query.squeeze()).float().mean()

        return {
            'SupervisedAcc_softmax': softmax_supervised_accuracy,
            'SupervisedAcc_sinkhorn': sinkhorn_supervised_accuracy,
            'SupervisedAcc_twostep': two_step_softmax_supervised_accuracy,
            'SupervisedLoss_softmax': softmax_supervised_loss,
            'SupervisedLoss_sinkhorn': sinkhorn_supervised_loss,
            'SupervisedLoss_twostep': two_step_softmax_supervised_loss,
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

    def clustering_loss(self, embedded_sample, regularization, clustering_type, sanity_check=True):
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

        z_support_flat = z_support.view(n_class*n_support, z_dim)
        z_query_flat = z_query.view(n_class*n_query, z_dim)

        # Class indices; usually 0, ..., n_class-1 unless sanity check
        class_indices = torch.arange(0, n_class)

        if sanity_check:  # Permute class labels and data for sanity check
            # step 1, assign random labels
            # Reassign the labels randomly
            label_reassignment = np.random.permutation(n_class)
            class_indices = torch.tensor(label_reassignment)

        target_inds_support = class_indices.view(n_class, 1, 1).expand(n_class, n_support, 1).flatten().long().to(z_support.device)
        target_inds_query = class_indices.view(n_class, 1, 1).expand(n_class, n_query, 1).flatten().long().to(z_support.device)

        if sanity_check:
            # step 2, permute data randomly
            support_permutation = np.random.permutation(n_class*n_support)
            query_permutation = np.random.permutation(n_class*n_query)

            z_support_flat = z_support_flat[support_permutation]
            z_query_flat = z_query_flat[query_permutation]

            target_inds_support = target_inds_support[support_permutation]
            target_inds_query = target_inds_query[query_permutation]


        # Build dummy targets (replace with gather maybe?)
        target_inds_dummy_support = np.zeros((n_class*n_support, n_class))
        target_inds_dummy_support[range(n_class*n_support), target_inds_support.cpu().numpy().flatten()] = 1. # transform targets to one-hot
        target_inds_dummy_support = torch.FloatTensor(target_inds_dummy_support).to(z_query.device)

        # Cluster support set into clusters (both for learning to cluster and unsupervised few shot learning)
        if clustering_type == 'wasserstein':
            z_centroid, __ = wasserstein.cluster_wasserstein(z_support_flat, n_class, stop_gradient=False, regularization=regularization)
        elif clustering_type == 'kmeans':
            z_centroid  = wasserstein.cluster_kmeans(z_support_flat, n_class, kmeansplusplus=False)
        elif clustering_type == 'kmeansplusplus':
            z_centroid  = wasserstein.cluster_kmeans(z_support_flat, n_class, kmeansplusplus=True)
        else:
            raise Exception('Clustering type not implemented {}'.format(clustering_type))

        # Pairwise distance from query set to centroids
        support_dists = euclidean_dist(z_support_flat, z_centroid)
        query_dists = euclidean_dist(z_query_flat, z_centroid)

        # Assign support set points to centroids (using either Sinkhorn or Softmax)
        all_log_p_y_support = {}
        ############ Sinkhorn conditionals ###################
        # Optimal assignment (could have kept previous result probably)
        __, __, log_assignment, __, __ = wasserstein.compute_sinkhorn_stable(
            support_dists, regularization=regularization, iterations=10)
        # Predictions are already the optimal assignment
        all_log_p_y_support['sinkhorn'] = log_assignment

        ############ Softmax conditionals ###################
        # Unpermuted Log Probabilities
        all_log_p_y_support['softmax'] = F.log_softmax(-support_dists*regularization, dim=1)


        ############ Make predictions in Few-shot clustering (Support) and Unsupervised Few shot learning (Query) mode ###################
        all_support_clustering_accuracy = {}
        all_query_clustering_accuracy = {}
        for conditional_mode, log_p_y_support in all_log_p_y_support.items():

            # Build accuracy permutation matrix (to match support with ground truth)
            # this could be cleaner using "gather"
            __, y_hat_support = log_p_y_support.max(1)
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

            # Now, run standard prototypical networks, but plugging centroids instead of prototypes
            log_p_y_query = F.log_softmax(-query_dists*regularization, dim=1)
            _, y_hat_query = log_p_y_query.max(1)
            y_hat_query = y_hat_query.cpu().numpy()

            # Permute predictions
            query_permuted_predictions = cols_support[y_hat_query]
            query_clustering_accuracy = (query_permuted_predictions == target_inds_query.cpu().numpy().flatten()).mean()

            all_support_clustering_accuracy[conditional_mode] = support_clustering_accuracy
            all_query_clustering_accuracy[conditional_mode] = query_clustering_accuracy

        return {
            'SupportClusteringAcc_softmax': all_support_clustering_accuracy['softmax'],
            'SupportClusteringAcc_sinkhorn': all_support_clustering_accuracy['sinkhorn'],
            'QueryClusteringAcc_softmax': all_query_clustering_accuracy['softmax'],
            'QueryClusteringAcc_sinkhorn': all_query_clustering_accuracy['sinkhorn']
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


# Load architecture used in the CCN paper
@register_model('ccn')
def load_ccn(**kwargs):
    x_dim = kwargs['x_dim']
    hid_dim = kwargs['hid_dim']
    z_dim = kwargs['z_dim']

    vggs = VGGS(2)  # it is assumed input of size 1x32x32

    encoder = nn.Sequential(
        vggs.features,
        Flatten()
    )

    return ClusterNet(encoder)
