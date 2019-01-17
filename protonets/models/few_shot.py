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

    def loss(self, sample, regularization, supervised_sinkhorn_loss, raw_input):
        xs = Variable(sample['xs']) # support
        xq = Variable(sample['xq']) # query

        n_class = xs.size(0)
        assert xq.size(0) == n_class
        n_support = xs.size(1)
        n_query = xq.size(1)

        target_inds = torch.arange(0, n_class).view(n_class, 1, 1).expand(n_class, n_query, 1).long()
        target_inds = Variable(target_inds, requires_grad=False)

        if xq.is_cuda:
            target_inds = target_inds.cuda()

        x = torch.cat([xs.view(n_class * n_support, *xs.size()[2:]),
                       xq.view(n_class * n_query, *xq.size()[2:])], 0)

        if raw_input:
            # No embedding, z = x
            z = x.view(len(x), -1)
        else:
            z = self.encoder.forward(x)
        z_dim = z.size(-1)

        z_proto = z[:n_class*n_support].view(n_class, n_support, z_dim).mean(1)
        z_proto_var = ((z_proto - z[:n_class*n_support].view(n_class, n_support, z_dim).mean(1)[:, None, :])**2).mean()
        zq = z[n_class*n_support:]

        dists = euclidean_dist(zq, z_proto)

        if not supervised_sinkhorn_loss:
            # Default protonet. Assignment is softmax on squared cluster-sample distance
            # divide/multiply by correct temperature/regularization
            log_p_y = F.log_softmax(-dists*regularization, dim=1).view(n_class, n_query, -1)
        else:
            # This part changes
            # Assignment is now regularized, optimal transport, but transportation cost is given by distance matrix.
            # this differs from previously because the regularization is slightly different
            # todo: grid search on iterations
            __, __, log_assignment, __, __ = wasserstein.compute_sinkhorn_stable(
                dists, regularization=regularization, iterations=10)
            log_p_y = log_assignment.view(n_class, n_query, -1)

        loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()

        _, y_hat = log_p_y.max(2)
        acc_val = torch.eq(y_hat, target_inds.squeeze()).float().mean()

        return loss_val, {
            'loss': loss_val.item(),
            'acc': acc_val.item(),
            'z_proto_var': z_proto_var
        }

    def eval_loss(self, sample):
        return self.loss(sample)


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

    def eval_loss(self, sample, regularization, supervised_sinkhorn_loss, raw_input):
        xs = Variable(sample['xs'])  # support
        xq = Variable(sample['xq'])  # query

        n_class = xs.size(0)
        assert xq.size(0) == n_class
        n_support = xs.size(1)
        n_query = xq.size(1)

        target_inds = torch.arange(0, n_class).view(n_class, 1, 1).expand(n_class, n_query, 1).long()
        target_inds = Variable(target_inds, requires_grad=False)

        if xq.is_cuda:
            target_inds = target_inds.cuda()

        x = torch.cat([xs.view(n_class * n_support, *xs.size()[2:]),
                       xq.view(n_class * n_query, *xq.size()[2:])], 0)

        if raw_input:
            # No embedding, z = x
            z = x.view(len(x), -1)
        else:
            z = self.encoder.forward(x)
        z_dim = z.size(-1)

        #z_proto = z[:n_class * n_support].view(n_class, n_support, z_dim).mean(1)

        zs = z[:n_class * n_support]
        zq = z[n_class * n_support:]

        # Cluster support set into clusters
        z_proto, data_centroid_assignment = wasserstein.cluster_wasserstein(zs, n_class, stop_gradient=False, regularization=regularization)

        # So it turns out the wasserstein assignments are not the same as the log_p_y assignments
        # one relies on optimal transport while the other relies on sample-centroid distances ...

        # Pairwise distance from query set to centroids
        dists = euclidean_dist(zq, z_proto)

        if supervised_sinkhorn_loss:
            # Optimal assignment (could have kept previous result probably)
            __, __, log_assignment, __, __ = wasserstein.compute_sinkhorn_stable(
                dists, regularization=regularization, iterations=10)
            log_p_y = log_assignment.view(n_class, n_query, -1)
        else:
            # Classic softmax

            # Unpermuted Log Probabilities
            log_p_y = F.log_softmax(-dists*regularization, dim=1).view(n_class, n_query, -1)



        target_inds_dummy = np.zeros((n_class*n_query, n_class))
        target_inds_dummy[range(n_class*n_query), target_inds.cpu().numpy().flatten()] = 1. # transform targets to one-hot
        target_inds_dummy = torch.FloatTensor(target_inds_dummy).to(xs.device)

        # Build permutation cost matrix
        permutation_cost = -log_p_y.view(n_class, n_query, n_class, 1) * target_inds_dummy.view(n_class, n_query, 1, n_class)
        permutation_cost = permutation_cost.sum(1).sum(0)

        # Compute best label assignment and corresponding loss
        # assignment[a,b] = 1_{a=\sigma(b)}
        # min_{\gamma} \sum_{a,b} permutation_cost[a,b] * assignment[a,b]
        # might not backprop through the graph though ...
        # it might or might not be equivalent due to the contraints (is it a critical point?)

        #loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()

        #log_p_y_permuted[n, k, a] = n_class * \sum_b log_p_y[n, k, b] * assignment[a,b]

        # Permuted log probabilities
        loss_val_unnormalized, assignment, __, __, __ = wasserstein.compute_sinkhorn_stable(
            permutation_cost, regularization=100., iterations=10)

        loss_val = loss_val_unnormalized / n_query  # normalize so it looks like a normal cross entropy

        log_p_y_permuted = n_class * torch.matmul(log_p_y, assignment)

        #_, y_hat = log_p_y.max(2)
        __, y_hat = log_p_y_permuted.max(2)

        acc_val = torch.eq(y_hat, target_inds.squeeze()).float().mean()

        # HUNGARIAN ALGORITHM
        # compute the same with hungarian
        #hungarian_loss, hungarian_assignment = wasserstein.compute_hungarian(permutation_cost)
        #print 'Hungarian', hungarian_loss
        #print 'Sinkhorn', loss_val

        __, argmax = log_p_y.view(n_class*n_query, n_class).max(1)
        argmax = argmax.cpu()  # no need to backprop anyways
        one_hot_prediction = torch.zeros((n_class*n_query, n_class)).to(xs.device)
        one_hot_prediction[range(n_class*n_query), argmax] = 1.
        accuracy_permutation_cost = -one_hot_prediction.view(n_class, n_query, n_class, 1) * target_inds_dummy.view(n_class, n_query, 1, n_class)
        accuracy_permutation_cost = accuracy_permutation_cost.sum(1).sum(0)

        __, __, cols = wasserstein.compute_hungarian(accuracy_permutation_cost)
        permuted_prediction = cols[argmax]
        clustering_accuracy = (permuted_prediction == target_inds.cpu().numpy().flatten()).mean()

        #print 'Clustering Accuracy', clustering_accuracy

        return loss_val, {
            'loss': loss_val.item(),
            '_ClusteringAccCE': acc_val.item(),
            'ClusteringAcc': clustering_accuracy
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
