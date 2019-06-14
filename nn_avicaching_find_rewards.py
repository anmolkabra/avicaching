#!/usr/bin/env python

# This script runs the pricing problem k-layered models and outputs log 
# files, and per-epoch plots. Also can be used to test other reward schemes on 
# learned identification problem models.

import argparse
import time
import os
import sys
import json
from functools import reduce
import numpy as np
import matplotlib
try:
    os.environ["DISPLAY"]
except KeyError as _:
    # working without X/GUI environment
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import avicaching_data as ad
# import torch modules
import torch
import torch.nn as nn
import torch.nn.functional as torchfun
import torch.optim as optim

# =============================================================================
# training options
# =============================================================================
parser = argparse.ArgumentParser(
    description="NN Avicaching model for finding rewards"
)
# training parameters
parser.add_argument("--data-settings-file", type=str,
                    metavar="DSF", default="./nn_avicaching_data_settings.json",
                    help="location of JSON file containing data file locations "
                    "for Avicaching models (default=\"./nn_avicaching_data_"
                    "settings.json\")")
parser.add_argument("--weights-file", required=True, type=str,
                    metavar="file",
                    help="inputs the location of the file to use weights from")
parser.add_argument("--lr", type=float,
                    metavar="LR", default=1e-3,
                    help="inputs learning rate of the network (default=1e-3)")
parser.add_argument("--eta", type=float,
                    metavar="F", default=1.0,
                    help="[see script] inputs parameter eta in the "
                    "model (default=1.0)")
parser.add_argument("--no-cuda",
                    action="store_true", default=False,
                    help="disables CUDA training")
parser.add_argument("--epochs", type=int,
                    metavar="E", default=1000,
                    help="inputs the number of epochs to train for (default=1000)")

# data options
parser.add_argument("--locations", type=int,
                    metavar="J", default=116,
                    help="inputs the number of locations (default=116)")
parser.add_argument("--time", type=int,
                    metavar="T", default=182,
                    help="inputs total time of data collection; number of "
                    "weeks (default=182)")
parser.add_argument("--rewards", type=float,
                    metavar="R", default=1000.0,
                    help="inputs the total budget of rewards to be distributed "
                    "(default=1000.0)")
parser.add_argument("--rand",
                    action="store_true", default=False,
                    help="uses random data")
parser.add_argument("--test", type=str,
                    metavar="TEST", default="",
                    help="inputs the location of the file to test rewards from "
                    "(default=\"\")")
parser.add_argument("--seed", type=int,
                    metavar="S", default=1,
                    help="seed (default=1)")

# plot/log options
parser.add_argument("--no-plots",
                    action="store_true", default=False,
                    help="does not plot any results")
parser.add_argument("--show-loss-plot",
                    action="store_true", default=False,
                    help="shows the loss plot")

# deprecated options -- not deleting if one chooses to use them
# if using SGD. Remember to check the model's optimizer in this file.
parser.add_argument("--momentum", type=float,
                    metavar="M", default=1.0,
                    help="DEPRECATED: [see script] inputs SGD momentum "
                    "(default=1.0)")

args = parser.parse_args()

# assigning cuda check and test check to single variables
args.cuda = not args.no_cuda and torch.cuda.is_available()

# set the seeds
torch.manual_seed(args.seed)
np.random.seed(seed=args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# =============================================================================
# parameters and constants
# =============================================================================

# global values and datasets
N_IDEN_PROB = 0         # layers used by iden prob
MORE_THAN_1_W = False   # True if more than 1 set of weights used by iden prob
J, T, R_TOTAL, num_features = args.locations, args.time, args.rewards, 0
J_avi = 0   # number of avicaching locations
WEIGHTS_FNAME = args.weights_file
X, F_DIST_w1, is_avi, not_avi_idx = [], [], [], []
back_loss = 0   # loss value actually backpropagated (not divided by mean(Y))
loss, l2_loss = 0, 0
w = {'first_for_r': [], 'except_first': []}

ORIG_DATA_FILES, RAND_DATA_FILES = ad.read_data_settings_file(
    args.data_settings_file
)

# misc settings and constants
matplotlib.rcParams.update({'font.size': 14})   # font-size for plots
WEIGHT_SAVE_FMT = "%.10f"
NP_DTYPE = np.float32

# =============================================================================
# data input
# =============================================================================

def read_FDIST():
    """
    Reads F and DIST data from the files, and combines them to make F_DIST,
    which gets weighed in the network. Helper to read_data_set().

    Returns:
        torch.Tensor -- F_DIST (combined F and DIST information ready)
    """
    global num_features, is_avi, not_avi_idx, J_avi

    F, is_avi = ad.read_F_file(
        RAND_DATA_FILES['F'] if args.rand else ORIG_DATA_FILES['F'], J
    )
    DIST = ad.read_dist_file(
        RAND_DATA_FILES['DIST'] if args.rand else ORIG_DATA_FILES['DIST'], J
    )

    # process data for the network
    F = ad.normalize(F, along_dim=0, using_max=True)
    DIST = ad.normalize(DIST, using_max=True)
    not_avi_idx = np.nonzero(is_avi - 1)[0]  # Indices of where is_avi is zero

    # combine f and D for the NN
    num_features = len(F[0]) + 1    # extra 1 for the distance element in F_DIST
    J_avi = int(sum(is_avi))
    F_DIST = torch.from_numpy(ad.combine_DIST_F(F, DIST, J, num_features))
    is_avi = torch.from_numpy(is_avi)
    not_avi_idx = torch.from_numpy(not_avi_idx)
    num_features += 1   # extra 1 for reward, that is added later

    return F_DIST

def read_X():
    """
    Reads X data from files. Helper to read_data_set().
    """
    global X

    X, _, _ = ad.read_XYR_file(
        RAND_DATA_FILES['XYR'] if args.rand else ORIG_DATA_FILES['XYR'], J, T
    )

    # condense X along T into a single vector and normalize
    X = ad.normalize(X.sum(axis=0), using_max=False)

    X = torch.from_numpy(X)

def read_weights(F_DIST):
    """
    Reads weight data from WEIGHTS_FNAME. Helper to read_data_set().

    Args:
        F_DIST -- (torch.Tensor) F_DIST combined data that will be multiplied
            by the first set of weights as optimized preprocessing
    """
    global F_DIST_w1, w, num_features, N_IDEN_PROB, MORE_THAN_1_W

    # read weights file first line, get json information into a dict
    with open(WEIGHTS_FNAME, "r") as f:
        WEIGHTS_FDATA = json.loads(f.readline())
        # has keys LAYERS, J, NUM_FEATURES

    # get and verify data from the weights file
    N_IDEN_PROB = WEIGHTS_FDATA['LAYERS']
    MORE_THAN_1_W = (N_IDEN_PROB > 2)  # more than 2 layers === more than 1 weight
    if J > WEIGHTS_FDATA['J']:
        raise RuntimeError("J in weights file is less than J provided to the "
                           "script; former >= latter")
    if WEIGHTS_FDATA['NUM_FEATURES'] != num_features:
        raise RuntimeError("num_features in weights file is not the same as "
                           "num_features provided to this script")

    # read weights file
    # no. of weights is 1 less than no. of layers in weights file
    list_of_w = ad.read_weights_file(
        WEIGHTS_FNAME, N_IDEN_PROB-1, WEIGHTS_FDATA['J'], J, num_features
    )
    list_of_w[-1] = np.expand_dims(list_of_w[-1], axis=2)

    # split w[0]; multiply the F_DIST portion of w[0] with F_DIST
    w1_for_fdist, w1_for_r = np.split(list_of_w[0], [num_features-1], axis=1)
    F_DIST_w1 = F_DIST.bmm(torch.from_numpy(w1_for_fdist))

    w['first_for_r'] = torch.from_numpy(w1_for_r)
    w['except_first'] = [torch.from_numpy(wi) for wi in list_of_w[1:]]

def read_data_set():
    """
    Reads Datasets X, Y, f, D, weights from the files using avicaching_data
    module's functions. f and D are then combined into F_DIST as preprocessed
    tensor, which is then multiplied with w1 as preprocessing. All datasets are
    normalized, expanded, averaged as required, leaving as torch tensors at the
    end of the function.
    """
    # shapes of datasets -- [] means expanded form:
    # - X: J
    # - net.R: J [x J x 1]
    # - F_DIST: J x J x num_features
    # - F_DIST_w1: J x J x num_features
    # - w['except_first'][-1]: (last weights) J x num_features [x 1]
    # - w['except_first'][1:-1]: (second to last weights) J x J x num_features
    # - first weights **were** also J x J x num_features
    # - w['first_for_r']: J x 1 x num_features

    read_X()
    read_weights(read_FDIST())

# =============================================================================
# forward network arch
# =============================================================================
def batch_norm_tensor(t):
    """
    Normalize tensor to 0 mean and 1 variance.
    """
    return (t - t.mean()) / torch.sqrt(t.var())

def forward_step_layer(t1, t2, activation_f=torchfun.relu):
    """
    Computes one step of input tensor t1 acted on tensor t2, emulating one
    layer multiplication in the forward step.

    Find this function in Identification Problem script for more information.
    """
    return batch_norm_tensor(activation_f(t1.bmm(t2)))

# =============================================================================
# PriProb class
# =============================================================================
class PriProb(nn.Module):
    """
    An instance of this class emulates the sub-model used in Pricing Problem to
    calculate the loss function and update rewards. Constraining not done here.
    """

    def __init__(self):
        """Initializes PriProb, creates the rewards dataset for the model."""
        super(PriProb, self).__init__()
        # initialize R: distribute R_TOTAL reward points in J_avi locations randomly
        # self.r preserved for debugging, no real use in the script
        self.r = np.array(ad.randint_upto_sum(R_TOTAL, J_avi)).astype(NP_DTYPE)

        # expand self.r from J_avi locations to J locations using is_avi
        self.r_exp = np.zeros((J), dtype=NP_DTYPE)
        self.r_exp[np.nonzero(is_avi.cpu().numpy())] = self.r

        #normalizedR = ad.normalize(self.r_exp, using_max=False)
        self.R = nn.Parameter(torch.from_numpy(self.r_exp))

    def forward(self, w):
        """
        Goes forward in the network -- multiply the weights, apply relu,
        multiply weights again and apply softmax

        Returns:
            torch.Tensor -- result after going forward in the network.
        """
        # Repeat the rewards as they were repeated at the end of F in
        # Identification problem (see build_input() in find_weights.py script
        repeatedR = self.R.repeat(J, 1).unsqueeze(dim=2)    # shape is J x J x 1

        # multiply w1 with r and add the resulting tensor with the already
        # calculated F_DIST_w1. Drastically improves performance.

        # If you've trouble understanding why multiply and add, draw these
        # tensors on a paper and work out how the additions and multiplications
        # affect elements, i.e., which operations affect which sections of the
        # tensors

        # res is J x J x num_features after
        # batch multiply repeatedR and w for r, then add F_DIST_w1
        res = torch.baddbmm(F_DIST_w1, repeatedR, w['first_for_r'])

        # forward propagation done, multiply remaining tensors (no tensors are
        # mutable after this point except res)
        # last w doesn't need relu
        if MORE_THAN_1_W:
            res = batch_norm_tensor(torchfun.relu(res))    # relu for weight 1
            # bmm -> relu for all but last weight set
            res = reduce(forward_step_layer, w['except_first'][:-1], res)
            res = res.bmm(w['except_first'][-1])
        res = res.view(-1, J)    # res is J x J
        res += eta_matrix

        return torchfun.softmax(res, dim=1)

def go_forward(net):
    """
    Feed forward the dataset in the model's network and calculate Y and loss.

    Args:
        net -- (PriProb instance)

    Returns:
        float -- time taken to go complete all operations in the network
    """
    global w, back_loss, loss, l2_loss
    start_forward_time = time.time()

    # feed in data
    P = net(w).t()

    # calculate loss
    Y = P.mv(X)
    Ybar = Y.mean()
    back_loss = (Y - Ybar).norm(1) / (J)
    loss = back_loss / Ybar
    l2_loss = ((Y - Ybar).norm(2) ** 2) / (J * Ybar)

    return time.time() - start_forward_time


def non_avi_zero_grad_hook(grad):
    """
    Returns a clone of gradients grad, calculated w.r.t. rewards, with
    those for non-avicaching locations set to 0. This hook must be registered
    before calling backward on loss.
    """
    grad_clone = grad.clone()
    grad_clone[not_avi_idx] = 0
    return grad_clone


def train(net, optimizer):
    """
    Trains the Neural Network using PriProb on the training set.

    Args:
        net -- (PriProb instance)
        optimizer -- (torch.optim instance) Gradient-Descent function

    Returns:
        3-tuple -- (Execution Time, End loss value,
            Model's prediction after feed forward [Px])
    """
    global back_loss

    # BACKPROPAGATE
    start_backprop_time = time.time()

    optimizer.zero_grad()
    back_loss.backward()         # calculate grad
    optimizer.step()        # update rewards

    backprop_time = time.time() - start_backprop_time

    # CONSTRAIN -- clamp minimum rewards at 0, then normalize using L1 norm
    constrain_time = time.time()
    net.R.data = torchfun.normalize(net.R.data.clamp(min=0), p=1, dim=0) * R_TOTAL
    constrain_time = time.time() - constrain_time

    # FORWARD
    forward_time = go_forward(net)

    return (backprop_time + constrain_time + forward_time, constrain_time)

# =============================================================================
# logs and plots
# =============================================================================
def save_log(fname, end_loss, end_l2_loss, t_time, rewards=None):
    """
    Saves the log to a file.

    Args:
        fname -- (str) name of the file (without the extension)
        end_loss -- (float) end loss value
        end_l2_loss -- (float) end L1 loss value
        t_time -- (float) runtime of model
        rewards -- (NumPy ndarray or None) rewards (default=None)
    """
    # write summary of logs, and the parameters used
    with open(fname + ".json", "wt") as f:
        j = {
            # data params
            'J': J,
            'T': T,
            'num_features': num_features,
            'R': R_TOTAL,
            'weights_file': WEIGHTS_FNAME,
            # model hyperparams
            'epochs': args.epochs,
            'layers': N_IDEN_PROB,
            'eta': args.eta,
            # misc params
            'seed': args.seed,
            # end results
            'runtime': t_time,
            'end_loss': float(end_loss),
            'end_l2_loss': float(end_l2_loss),
        }

        if args.test:
            j['test_file'] = args.test

        if args.rand:
            j['device'] = 'gpu' if args.cuda else 'cpu'

        json.dump(j, f, indent=4, separators=(',', ': '))

    if rewards is not None:
        with open(fname + ".txt", "wt") as f:
            np.savetxt(f, np.expand_dims(rewards, axis=0), fmt="%.15f", delimiter=" ")

def save_plot(fname, x, y, xlabel, ylabel, title):
    """
    Saves and (optionally) shows the loss plot of train and test periods.

    Args:
        fname -- (str) name of the file (without the extension)
        x -- (NumPy ndarray) data on the x-axis
        y -- (NumPy ndarray) data on the y-axis
        xlabel -- (str) label for the x-axis
        ylabel -- (str) what else can it mean?
        title -- (str) title of the plot
    """
    # plot details
    loss_fig = plt.figure(1)
    plt.plot(x, y, "r-", label="Train Loss")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, which="major", axis="both", color="k", ls="dotted", lw="1.0")
    plt.grid(True, which="minor", axis="y", color="k", ls="dotted", lw="0.5")
    plt.minorticks_on()
    plt.title(title)

    # save and show
    loss_fig.tight_layout()
    loss_fig.savefig(fname + ".png", bbox_inches="tight", dpi=200)
    if args.show_loss_plot:
        plt.show()
    plt.close()

# =============================================================================
# main program
# =============================================================================
if __name__ == "__main__":
    read_data_set()
    net = PriProb()

    # Attach hook to set the non-avicaching gradients for rewards to zero so that
    # the non-avicaching rewards do not update
    net.R.register_hook(non_avi_zero_grad_hook)

    eta_matrix = args.eta * torch.eye(J)
    transfer_time = time.time()
    if args.cuda:
        net.cuda()
        F_DIST_w1, X = F_DIST_w1.cuda(), X.cuda()
        w['first_for_r'] = w['first_for_r'].cuda()
        w['except_first'] = [wi.cuda() for wi in w['except_first']]
        is_avi = is_avi.cuda()
        not_avi_idx = not_avi_idx.cuda()
        eta_matrix = eta_matrix.cuda()
        file_pre_gpu = "gpu, "
    else:
        file_pre_gpu = "cpu, "
    transfer_time = time.time() - transfer_time
    total_time = transfer_time

    if args.test:
        # secondary function of the script -- calculate loss value for the
        # supplied data
        rewards = np.loadtxt(args.test, delimiter=" ", dtype=NP_DTYPE)[:J]
        R_TOTAL = float(np.sum(rewards))
        rewards = torch.from_numpy(ad.normalize(rewards, using_max=False))
        if args.cuda:
            rewards = rewards.cuda()
        net.R.data = rewards        # substitute manual rewards
        forward_time = go_forward(net)
        # save results
        fname = 'testing "' + args.test[args.test.rfind("/") + 1:] + '" ' + str(time.time())
        save_log("./stats/find_rewards/test_rewards_results/" + fname,
                 loss.data, l2_loss.data, forward_time)
        sys.exit(0)

    # optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)
    optimizer = optim.Adam(net.parameters(), lr=args.lr)

    # refer to the report and Algorithm for Pricing Problem to understand the
    # logic flow: forward -> loop [backpropagate -> update -> constrain -> forward]
    # The loop is the way it is since forward weights the rewards and calculates
    # the loss.
    total_constraint_time = 0
    total_time += go_forward(net)    # start model and logging here
    train_loss = [loss.data]
    best_loss, best_rew = loss.data, net.R.data.clone()
    best_l2_loss = l2_loss.data

    for e in range(args.epochs):
        train_t = train(net, optimizer)
        curr_loss = loss.data
        train_loss.append(curr_loss)

        if curr_loss < best_loss:
            # save the best result uptil now
            best_loss = curr_loss
            best_l2_loss = l2_loss.data
            best_rew = net.R.data.clone()

        total_time += train_t[0]
        total_constraint_time += train_t[1]
        if e % 20 == 0:
            print("epoch={0:5d}, loss={1:0.10f}, budget={2:0.10f}".format(
                e, curr_loss, net.R.data.sum()
            ))
    total_epochs = args.epochs

    best_rew = best_rew.cpu().numpy()

    # log and plot the results: epoch vs loss

    # define file names
    file_pre = "{0:s}XYR_seed={1:d}, epochs={2:d}, ".format(
        "rand" if args.rand else "orig", args.seed, total_epochs
    )
    constraint_suff = "constr_time={0:0.4f}".format(total_constraint_time)
    log_name = "lr={0:0.3e}, bestloss={1:0.6f}, time={2:0.4f} sec, ".format(
        args.lr, best_loss, total_time
    )
    epoch_data = np.arange(total_epochs+1)  # extra epoch for the first forward pass
    fname = "{0:d}layer_".format(N_IDEN_PROB) + file_pre_gpu + file_pre + log_name + \
            constraint_suff
    # save amd plot data
    save_log("./stats/find_rewards/logs/" + fname, best_loss, best_l2_loss, total_time,
             best_rew)
    if not args.no_plots:
        save_plot("./stats/find_rewards/plots/" + fname, epoch_data,
                  train_loss, "epoch", "loss", log_name)

    print("---> " + fname + " DONE")
