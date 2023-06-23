from torch.utils.data.sampler import Sampler
import datetime
import datetime as dt
import sys
import warnings
import dgl
import argparse
import math
import time

import numpy as np
import pandas as pd
import torch as th
import torch.nn.functional as F
import torch.optim as optim
import wandb
import os
import gc
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator

import dgl.nn.pytorch as dglnn
import torch
import torch.nn as nn
from dgl import function as fn
from dgl.ops import edge_softmax
from dgl.utils import expand_as_pair

device = None
in_feats, n_classes = None, None
epsilon = 1 - math.log(2)


warnings.filterwarnings("ignore")

dgl.random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.set_num_threads(64)


class YearWiseRandomSampler(Sampler):
    def __init__(self, g, mask, batch_size, is_train):
        self.g = g
        self.batch_size = batch_size
        self.mask = mask
        self.mask_one = (mask != 0).nonzero().squeeze()
        self.is_train = is_train

    def __iter__(self):
        timestamps = self.g.ndata["year"].long().reshape(-1)[self.mask_one].numpy()
        if self.is_train:
            random_years = np.random.permutation(np.unique(timestamps))
        else:
            random_years = np.sort(np.unique(timestamps))
        for random_year in random_years:
            sampled = np.where([day == random_year for day in timestamps])[0]
            if self.is_train:
                indices = torch.randperm(len(sampled))
            else:
                indices = torch.Tensor(np.arange(len(sampled))).long()
            num_batches = len(sampled) // self.batch_size
            if self.is_train:
                sampled = np.where([day == random_year for day in timestamps])[0]
            else:
                sampled = np.sort(
                    np.where([day == random_year for day in timestamps])[0]
                )

            if len(sampled) == 1 and self.is_train == True:
                continue

            for i in range(0, num_batches + 1):
                start = i * self.batch_size
                end = min((i + 1) * self.batch_size, len(sampled))
                if end - start != self.batch_size and self.is_train == True:
                    continue
                if end - start == 1 and self.is_train == False:
                    yield [sampled[indices[start:end]]], random_year
                    continue
                yield sampled[indices[start:end]], random_year

    def __len__(self):
        return len(self.data_source)


class TimeRelaxGraphNodeSampler:
    def __init__(self, g, mask, samplerParams, isTrain):
        self.g = g
        self.fanouts = samplerParams["fanout"]
        self.nLayers = samplerParams["nhops"]
        self.isTrain = isTrain
        self.dataloader = iter(self.get_dataloader(mask, samplerParams["batch_size"]))
        # find the indices where label information is available
        self.mask_nonzero_indices = (mask != 0).nonzero().squeeze()
        if self.isTrain:
            self.len = int(len(self.mask_nonzero_indices) / samplerParams["batch_size"])
        else:
            if len(self.mask_nonzero_indices) % samplerParams["batch_size"] > 0:
                self.len = (
                    int(len(self.mask_nonzero_indices) / samplerParams["batch_size"])
                    + 1
                )
            else:
                self.len = int(
                    len(self.mask_nonzero_indices) / samplerParams["batch_size"]
                )
        self.ibatch = 0

    def get_dataloader(self, mask, batch_size):
        if self.isTrain:
            dataloader = YearWiseRandomSampler(self.g, mask, batch_size, self.isTrain)
        else:
            dataloader = YearWiseRandomSampler(self.g, mask, batch_size, self.isTrain)
        return dataloader

    def getLength(self):
        return self.len

    def reIndex(self, edgesForBlock, dstNodesGlobal):
        allSrc = list(edgesForBlock.src.unique())
        srcDf = pd.DataFrame({"src": allSrc, "srcIx": [x for x in range(len(allSrc))]})
        allDst = dstNodesGlobal.numpy().tolist()
        dstDf = pd.DataFrame({"dst": allDst, "dstIx": [x for x in range(len(allDst))]})
        edatas = {}
        edgesForBlock = edgesForBlock.merge(srcDf, on="src", how="left")
        edgesForBlock = edgesForBlock.merge(dstDf, on="dst", how="left")

        edgesForBlock = edgesForBlock[["srcIx", "dstIx"]].drop_duplicates()
        edgesForBlockDict = {}
        tempEdges = edgesForBlock.copy()
        if tempEdges.shape[0] > 0:
            edgesForBlockDict = (tempEdges.srcIx.tolist(), tempEdges.dstIx.tolist())
        else:
            edgesForBlockDict = ([], [])
        secondaryBlock = dgl.create_block(
            edgesForBlockDict, num_src_nodes=len(allSrc), num_dst_nodes=len(allDst)
        )
        secondaryBlock.srcdata[dgl.NID] = torch.tensor(allSrc)
        secondaryBlock.dstdata[dgl.NID] = dstNodesGlobal
        return secondaryBlock

    def sample_block(self):
        blocks = []
        samplededges = []
        dataloader_next = next(self.dataloader)
        output_nids = self.mask_nonzero_indices[dataloader_next[0]]
        day_sampled = dataloader_next[1]
        cur = torch.LongTensor(output_nids)
        for i in range(self.nLayers):
            fanout = self.fanouts
            if fanout is None:
                frontier = dgl.in_subgraph(self.g, cur)
            else:
                frontier = dgl.sampling.sample_neighbors(self.g, cur, fanout)
            samplededges.insert(0, frontier.edata[dgl.EID])
            primaryBlock = dgl.to_block(frontier, cur, include_dst_in_src=False)
            batch_min_timestamp = day_sampled
            cur = {
                ntype: primaryBlock.srcnodes[ntype].data[dgl.NID]
                for ntype in primaryBlock.srctypes
            }
            frontier_src_to_dst = dgl.sampling.sample_neighbors(
                self.g, cur, -1, edge_dir="out"
            )
            frontier_dst_to_src = dgl.reverse(
                frontier_src_to_dst, copy_ndata=True, copy_edata=False
            )
            block_dst_to_src = dgl.to_block(
                frontier_dst_to_src,
                {
                    ntype: primaryBlock.srcnodes[ntype].data[dgl.NID]
                    for ntype in primaryBlock.srctypes
                },
                include_dst_in_src=False,
            )
            edgesForBlock = []
            edges = block_dst_to_src.all_edges()

            srcGlobal = block_dst_to_src.srcdata[dgl.NID].numpy().tolist()
            dstGlobal = block_dst_to_src.dstdata[dgl.NID].numpy().tolist()

            srcLocal = edges[0].numpy().tolist()
            dstLocal = edges[1].numpy().tolist()
            srcOut = [srcGlobal[x] for x in srcLocal]
            dstOut = [dstGlobal[x] for x in dstLocal]

            srcGlobalTimestamp = (
                self.g.ndata["year"][srcOut].numpy().reshape(-1).tolist()
            )
            edgePrimaryDf = pd.DataFrame(
                {"src": srcOut, "dst": dstOut, "srctime": srcGlobalTimestamp}
            )
            edgePrimaryDf.index = edgePrimaryDf.srctime
            edgePrimaryDf = edgePrimaryDf[
                edgePrimaryDf.srctime <= batch_min_timestamp
            ].copy()
            edgePrimaryDf = edgePrimaryDf[["src", "dst"]].copy()
            edgesForBlock.append(edgePrimaryDf)
            edgesForBlock = pd.concat(edgesForBlock)
            secondaryBlock = self.reIndex(edgesForBlock, primaryBlock.srcdata[dgl.NID])
            secondaryBlock.srcdata["feat"] = self.g.ndata["feat"][
                secondaryBlock.srcdata[dgl.NID]
            ]
            secondaryBlock.dstdata["feat"] = self.g.ndata["feat"][
                secondaryBlock.dstdata[dgl.NID]
            ]

            secondaryBlock.srcdata["year"] = self.g.ndata["year"][
                secondaryBlock.srcdata[dgl.NID]
            ]
            secondaryBlock.dstdata["year"] = self.g.ndata["year"][
                secondaryBlock.dstdata[dgl.NID]
            ]

            blocks.insert(0, [secondaryBlock, primaryBlock])
            cur = {
                ntype: primaryBlock.srcnodes[ntype].data[dgl.NID]
                for ntype in primaryBlock.srctypes
            }
        return blocks, cur, output_nids, day_sampled


class YearWiseRandomSamplerReplay(Sampler):
    def __init__(self, g, mask, sampledBlocks, batch_size, is_train):
        self.g = g
        self.batch_size = batch_size
        self.mask = mask
        self.mask_one = (mask != 0).nonzero().squeeze()
        self.is_train = is_train
        self.sampledBlocks = sampledBlocks

    def __iter__(self):
        num_batches = len(self.mask_one) // self.batch_size
        timestamps = list(self.sampledBlocks.keys())
        if self.is_train:
            random_days = np.random.permutation(np.unique(timestamps))
        else:
            random_days = np.sort(np.unique(timestamps))
        for random_day in random_days:
            sampled = self.sampledBlocks[random_day]
            if self.is_train:
                indices = torch.randperm(len(sampled))
            else:
                indices = torch.Tensor(np.arange(len(sampled))).long()
            num_batches = len(indices)
            for i in indices:
                yield sampled[i], random_day

    def __len__(self):
        return len(self.data_source)


class TimeRelaxGraphNodeSamplerReplay:
    def __init__(self, g, mask, sampledBlocks, samplerParams, isTrain):
        self.g = g
        self.fanouts = samplerParams["fanout"]
        self.nLayers = samplerParams["nhops"]
        self.isTrain = isTrain
        self.sampledBlocks = sampledBlocks
        self.dataloader = iter(self.get_dataloader(mask, samplerParams["batch_size"]))
        self.mask_nonzero_indices = (mask != 0).nonzero().squeeze()

    def get_dataloader(self, mask, batch_size):
        if self.isTrain:
            dataloader = YearWiseRandomSamplerReplay(
                self.g, mask, self.sampledBlocks, batch_size, self.isTrain
            )
        else:
            dataloader = YearWiseRandomSamplerReplay(
                self.g, mask, self.sampledBlocks, batch_size, self.isTrain
            )
        return dataloader

    def getLength(self):
        return self.len

    def sample_block(self):
        return next(self.dataloader)


def buildSampler(graph, mask, samplerParams, isTrain, sampledBlocks=None):
    if samplerParams["type"] == "time_relax_sampler":
        sampler = TimeRelaxGraphNodeSampler(graph, mask, samplerParams, isTrain)
    elif samplerParams["type"] == "time_relax_sampler_replay":
        sampler = TimeRelaxGraphNodeSamplerReplay(
            graph, mask, sampledBlocks, samplerParams, isTrain
        )
    return sampler


class GCN(nn.Module):
    def __init__(
        self, in_feats, n_hidden, n_classes, n_layers, activation, dropout, use_linear
    ):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.use_linear = use_linear

        self.convs = nn.ModuleList()
        self.convs_backward = nn.ModuleList()
        self.convs_forward = nn.ModuleList()
        self.linear_after_combination = nn.ModuleList()

        if use_linear:
            self.linear = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(n_layers):
            in_hidden = n_hidden if i > 0 else in_feats
            out_hidden = n_hidden if i < n_layers - 1 else n_classes
            bias = i == n_layers - 1

            # For addition, max operation
            self.convs_backward.append(
                dglnn.GraphConv(
                    in_feats, in_hidden, "both", bias=bias, allow_zero_in_degree=True
                )
            )
            self.convs.append(
                dglnn.GraphConv(
                    in_hidden, out_hidden, "both", bias=bias, allow_zero_in_degree=True
                )
            )

            # self.convs_backward.append(dglnn.GraphConv(
            #     in_feats, in_hidden, "both", bias=bias))
            # self.convs.append(dglnn.GraphConv(
            #     in_hidden, out_hidden, "both", bias=bias))

            #     For concat operation
            # self.convs_backward.append(dglnn.GraphConv(
            #     in_feats, n_hidden, "both", bias=bias, allow_zero_in_degree=True))
            # self.convs.append(dglnn.GraphConv(
            #     in_hidden + n_hidden, out_hidden, "both", bias=bias, allow_zero_in_degree=True))

            #     For concat->linear operation
            # self.convs_backward.append(dglnn.GraphConv(
            #     in_feats, n_hidden, "both", bias=bias))
            # self.linear_after_combination.append(nn.Linear(in_hidden + n_hidden, n_hidden))
            # self.convs.append(dglnn.GraphConv(
            #     n_hidden, out_hidden, "both", bias=bias))

            #     For GCN addition, max operation
            # self.convs_backward.append(dglnn.GraphConv(
            #     in_feats, in_hidden, "both", bias=bias))
            # self.convs_forward.append(dglnn.GraphConv(
            #     in_hidden, out_hidden, "both", bias=bias))
            # self.convs.append(dglnn.GraphConv(
            #     in_hidden, out_hidden, "both", bias=bias))

            if use_linear:
                self.linear.append(nn.Linear(in_hidden, out_hidden, bias=False))
            if i < n_layers - 1:
                self.norms.append(nn.BatchNorm1d(out_hidden))

        self.input_drop = nn.Dropout(min(0.1, dropout))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

        self.input_drop_backward = nn.Dropout(min(0.1, dropout))

    def forward(self, graph, feat):
        h = feat
        h = self.input_drop(h)

        for i in range(self.n_layers):
            h_backward = self.input_drop_backward(graph[i][0].srcdata["feat"])
            h_backward = self.convs_backward[i](graph[i][0], h_backward)

            # h = h + h_backward
            # h = th.cat([h, h_backward], dim=1)
            h = th.max(h, h_backward)
            # h = self.linear_after_combination[i](th.cat([h, h_backward], dim=1))
            # h = th.cat([th.min(h, h_backward), th.max(h, h_backward)], dim=1)
            # h = th.min(h, h_backward) + th.max(h, h_backward)

            conv = self.convs[i](graph[i][1], h)

            if self.use_linear:
                linear = self.linear[i](h)
                h = conv + linear
            else:
                h = conv

            if i < self.n_layers - 1:
                h = self.norms[i](h)
                h = self.activation(h)
                h = self.dropout(h)

        return h


def gen_model(args):
    if args.use_labels:
        model = GCN(
            in_feats + n_classes,
            args.n_hidden,
            n_classes,
            args.n_layers,
            F.relu,
            args.dropout,
            args.use_linear,
        )
    else:
        model = GCN(
            in_feats,
            args.n_hidden,
            n_classes,
            args.n_layers,
            F.relu,
            args.dropout,
            args.use_linear,
        )
    return model


def cross_entropy(x, labels):
    y = F.cross_entropy(x, labels[:, 0], reduction="none")
    y = th.log(epsilon + y) - math.log(epsilon)
    return th.mean(y)


def compute_acc(pred, labels, evaluator):
    return evaluator.eval(
        {"y_pred": pred.argmax(dim=-1, keepdim=True), "y_true": labels}
    )["acc"]


def add_labels(feat, labels, idx):
    onehot = th.zeros([feat.shape[0], n_classes]).to(device)
    onehot[idx, labels[idx, 0]] = 1
    return th.cat([feat, onehot], dim=-1)


def adjust_learning_rate(optimizer, lr, epoch):
    if epoch <= 50:
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr * epoch / 50


def train(model, graph, labels, output_nids, optimizer, use_labels):
    model.train()
    feat = graph[0][1].srcdata["feat"]

    optimizer.zero_grad()
    pred = model(graph, feat)
    loss = cross_entropy(pred, labels[output_nids])
    loss.backward()
    optimizer.step()

    return loss, pred


@th.no_grad()
def evaluate(
    model, graph, labels, test_idx, args, evaluator, testSampler, filename, epoch
):
    model.eval()

    feat = graph.ndata["feat"]

    testrawpreds = torch.FloatTensor()
    output_nids_list = torch.FloatTensor()
    testloss = 0
    n_samples = 0

    while True:
        try:
            node_flow, input_nids, output_nids = testSampler.sample_block()[0]
            pred = model(node_flow, node_flow[0][1].srcdata["feat"])
            testloss += (
                cross_entropy(pred, labels[output_nids])
                * node_flow[-1][1].num_dst_nodes()
            )
            n_samples += node_flow[-1][1].num_dst_nodes()
            testrawpreds = torch.cat((testrawpreds, pred), dim=0)
            output_nids_list = torch.cat((output_nids_list, output_nids), dim=0)
        except StopIteration:
            break

    testloss = testloss / n_samples

    df = pd.DataFrame(
        {
            "id": output_nids_list.cpu().numpy(),
            "label": labels[output_nids_list.long()].cpu().numpy().reshape(-1),
            "pred": testrawpreds.argmax(dim=-1).cpu().numpy(),
        }
    )

    if args.use_labels:
        feat = add_labels(feat, labels, test_idx)

    return (
        compute_acc(testrawpreds, labels[output_nids_list.long()], evaluator),
        testloss,
    )


def get_sampledBlocks(args, graph, train_mask, isTrain):
    samplerParams = {
        "fanout": -1,
        "nhops": args.n_layers,
        "batch_size": args.batch_size,
        "type": "time_relax_sampler",
    }
    trainSampler = buildSampler(graph, train_mask, samplerParams, isTrain)
    sampledBlocks_Train = {}
    n = 0
    while True:
        try:
            n += 1
            if n % 10 == 0:
                print("done with n = ", n)
            (
                node_flow,
                input_nids,
                output_nids,
                sampled_day,
            ) = trainSampler.sample_block()
            if sampled_day not in list(sampledBlocks_Train.keys()):
                sampledBlocks_Train[sampled_day] = []
            sampledBlocks_Train[sampled_day].append(
                (node_flow, input_nids, output_nids)
            )
            _ = gc.collect()
        except StopIteration:
            break
    for key, val in sampledBlocks_Train.items():
        print(key, len(val))
    return sampledBlocks_Train


def run(args, graph, labels, train_idx, val_idx, test_idx, evaluator, n_running):
    # define model and optimizer
    model = gen_model(args)
    model = model.to(device)

    if args.debug:
        wandb.watch(model)

    print(model)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=100, verbose=True, min_lr=1e-3
    )

    # training loop
    total_time = 0
    best_val_acc, final_test_acc, best_val_loss = 0, 0, float("inf")

    train_accs, val_accs, test_accs = [], [], []
    train_losses, val_losses, test_losses = [], [], []

    train_mask = torch.zeros_like(labels).float().view(-1)
    train_mask[train_idx] = 1
    val_mask = torch.zeros_like(labels).float().view(-1)
    val_mask[val_idx] = 1
    test_mask = torch.zeros_like(labels).float().view(-1)
    test_mask[test_idx] = 1

    # Training replay sampledBlocks
    print("Sampling blocks for training")
    sampledBlocks_Train = get_sampledBlocks(args, graph, train_mask, True)

    # Evaluation replay sampledBlocks
    ## Train
    print("Sampling blocks for Train evaluation")
    sampledBlocks_evaluation_Train = get_sampledBlocks(args, graph, train_mask, False)
    ## Val
    print("Sampling blocks for Val evaluation")
    sampledBlocks_evaluation_Val = get_sampledBlocks(args, graph, val_mask, False)
    ## Test
    print("Sampling blocks for Test evaluation")
    sampledBlocks_evaluation_Test = get_sampledBlocks(args, graph, test_mask, False)

    for epoch in range(1, args.n_epochs + 1):
        tic = time.time()

        samplerParams = {
            "fanout": -1,
            "nhops": args.n_layers,
            "batch_size": args.batch_size,
            "type": "time_relax_sampler_replay",
        }
        trainSampler_replay = buildSampler(
            graph, train_mask, samplerParams, True, sampledBlocks_Train
        )

        while True:
            try:
                node_flow, input_nids, output_nids = trainSampler_replay.sample_block()[
                    0
                ]
                adjust_learning_rate(optimizer, args.lr, epoch)
                loss, pred = train(
                    model, node_flow, labels, output_nids, optimizer, args.use_labels
                )
            except StopIteration:
                break

        evaluation_trainSampler_replay = buildSampler(
            graph, train_mask, samplerParams, False, sampledBlocks_evaluation_Train
        )
        evaluation_valSampler_replay = buildSampler(
            graph, val_mask, samplerParams, False, sampledBlocks_evaluation_Val
        )
        evaluation_testSampler_replay = buildSampler(
            graph, test_mask, samplerParams, False, sampledBlocks_evaluation_Test
        )

        train_acc, train_loss = evaluate(
            model,
            graph,
            labels,
            train_idx,
            args,
            evaluator,
            evaluation_trainSampler_replay,
            "train",
            epoch,
        )
        val_acc, val_loss = evaluate(
            model,
            graph,
            labels,
            val_idx,
            args,
            evaluator,
            evaluation_valSampler_replay,
            "val",
            epoch,
        )
        test_acc, test_loss = evaluate(
            model,
            graph,
            labels,
            test_idx,
            args,
            evaluator,
            evaluation_testSampler_replay,
            "test",
            epoch,
        )

        lr_scheduler.step(loss)

        toc = time.time()
        total_time += toc - tic

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            final_test_acc = test_acc

        if epoch % args.log_every == 0:
            print(
                f"Run: {n_running}/{args.n_runs}, Epoch: {epoch}/{args.n_epochs}, Average epoch time: {total_time / epoch:.2f}\n"
                f"Train/Val/Test loss: {train_loss:.4f}/{val_loss:.4f}/{test_loss:.4f}\n"
                f"Train/Val/Test/Best val/Final test acc: {train_acc:.4f}/{val_acc:.4f}/{test_acc:.4f}/{best_val_acc:.4f}/{final_test_acc:.4f}"
            )

            if args.debug:
                wandb.log(
                    {
                        "Epoch": epoch,
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "test_loss": test_loss,
                        "train_acc": train_acc,
                        "val_acc": val_acc,
                        "test_acc": test_acc,
                        "best_val_acc": best_val_acc,
                        "final_test_acc": final_test_acc,
                    }
                )

        for l, e in zip(
            [train_accs, val_accs, test_accs, train_losses, val_losses, test_losses],
            [train_acc, val_acc, test_acc, train_loss, val_loss, test_loss],
        ):
            l.append(e)

    print("*" * 50)
    print(f"Best val acc: {best_val_acc}, Final test acc: {final_test_acc}")
    print("*" * 50)

    return best_val_acc, final_test_acc


def count_parameters(args):
    model = gen_model(args)
    return sum([np.prod(p.size()) for p in model.parameters() if p.requires_grad])


def main():
    global device, in_feats, n_classes

    argparser = argparse.ArgumentParser(
        "GCN on OGBN-Arxiv", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    argparser.add_argument(
        "--cpu", action="store_true", help="CPU mode. This option overrides --gpu."
    )
    argparser.add_argument("--gpu", type=int, default=0, help="GPU device ID.")
    argparser.add_argument("--n-runs", type=int, default=1, help="running times")
    argparser.add_argument(
        "--n-epochs", type=int, default=1000, help="number of epochs"
    )
    argparser.add_argument(
        "--use-labels",
        action="store_true",
        help="Use labels in the training set as input features.",
    )
    argparser.add_argument(
        "--use-linear", action="store_true", help="Use linear layer."
    )
    argparser.add_argument("--lr", type=float, default=0.005, help="learning rate")
    argparser.add_argument("--n-layers", type=int, default=3, help="number of layers")
    argparser.add_argument(
        "--n-hidden", type=int, default=256, help="number of hidden units"
    )
    argparser.add_argument("--dropout", type=float, default=0.5, help="dropout rate")
    argparser.add_argument("--wd", type=float, default=0, help="weight decay")
    argparser.add_argument(
        "--log-every", type=int, default=1, help="log every LOG_EVERY epochs"
    )
    argparser.add_argument(
        "--batch-size", type=int, default=1024, help="batch size used for training"
    )
    argparser.add_argument(
        "--debug", action="store_false", help="use this to dont log on wandb"
    )
    argparser.add_argument(
        "--name",
        type=str,
        default="trd-gcn",
        help="Name of the experiment",
    )
    args = argparser.parse_args()

    if args.debug:
        wandb.init(project="timerelax", name=args.name, config=args)
    print(args)

    if args.cpu:
        device = th.device("cpu")
    else:
        device = th.device("cuda:%d" % args.gpu)

    # load data
    data = DglNodePropPredDataset(name="ogbn-arxiv")
    evaluator = Evaluator(name="ogbn-arxiv")

    splitted_idx = data.get_idx_split()
    train_idx, val_idx, test_idx = (
        splitted_idx["train"],
        splitted_idx["valid"],
        splitted_idx["test"],
    )
    graph, labels = data[0]
    graph = graph.reverse(share_ndata=True, share_edata=True)

    # add reverse edges
    # srcs, dsts= graph.all_edges()
    # graph.add_edges(dsts, srcs)

    # """  Fix Inconsistencies in direction   """
    # year_srcs= graph.ndata["year"][srcs].cpu().numpy().reshape(-1)
    # year_dsts= graph.ndata["year"][dsts].cpu().numpy().reshape(-1)

    # inconsistency= np.where(year_srcs > year_dsts)[0]

    # graph.remove_edges(inconsistency)
    # graph.add_edges(dsts[inconsistency], srcs[inconsistency])

    # srcs, dsts = graph.all_edges()
    # year_srcs = graph.ndata["year"][srcs].cpu().numpy().reshape(-1)
    # year_dsts = graph.ndata["year"][dsts].cpu().numpy().reshape(-1)
    # inconsistency = np.where(year_srcs > year_dsts)[0]

    # add self-loop
    print(f"Total edges before adding self-loop {graph.number_of_edges()}")
    graph = graph.remove_self_loop().add_self_loop()
    print(f"Total edges after adding self-loop {graph.number_of_edges()}")

    in_feats = graph.ndata["feat"].shape[1]
    n_classes = (labels.max() + 1).item()
    graph.create_formats_()

    train_idx = train_idx.to(device)
    val_idx = val_idx.to(device)
    test_idx = test_idx.to(device)
    labels = labels.to(device)
    graph = graph.to(device)

    # run
    val_accs = []
    test_accs = []

    for i in range(args.n_runs):
        val_acc, test_acc = run(
            args, graph, labels, train_idx, val_idx, test_idx, evaluator, i
        )
        val_accs.append(val_acc)
        test_accs.append(test_acc)

    print(f"Runned {args.n_runs} times")
    print("Val Accs:", val_accs)
    print("Test Accs:", test_accs)
    print(f"Average val accuracy: {np.mean(val_accs)} ± {np.std(val_accs)}")
    print(f"Average test accuracy: {np.mean(test_accs)} ± {np.std(test_accs)}")
    print(f"Number of params: {count_parameters(args)}")


if __name__ == "__main__":
    main()
