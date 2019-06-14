#!/usr/bin/env python

# This script runs the identification problem k-layered models and outputs log 
# files, per-epoch plots, and model weights.

import argparse
import warnings
import time
import math
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
    description="NN Avicaching model for finding weights"
)
# training parameters
parser.add_argument("--layers", required=True, type=int,
                    metavar="N",
                    help="number of layers in the model's network (including in "
                    "and out layers")
parser.add_argument("--data-settings-file", type=str,
                    metavar="DSF", default="./nn_avicaching_data_settings.json",
                    help="location of JSON file containing data file locations "
                    "for Avicaching models (default=\"./nn_avicaching_data_"
                    "settings.json\")")
parser.add_argument("--lr", type=float,
                    metavar="LR", default=1e-3,
                    help="inputs learning rate of the network (default=1e-3)")
parser.add_argument("--eta", type=float,
                    metavar="F", default=1.0,
                    help="[see script] inputs parameter eta in the "
                    "model (default=1.0)")
parser.add_argument("--lambda-L2", type=float,
                    metavar="LAM", default=10.0,
                    help="[see script] inputs the L2 regularizing "
                    "coefficient (default=10.0)")
parser.add_argument("--no-cuda",
                    action="store_true", default=False,
                    help="disables CUDA training")
parser.add_argument("--record-test-res",
                    action="store_true", default=False,
                    help="enables the script to record test results")
parser.add_argument("--epochs", type=int,
                    metavar="E", default=1000,
                    help="inputs the number of epochs to train for (default=1000)")

# data options
parser.add_argument("--train-frac", type=float,
                    metavar="TF", default=0.75,
                    help="breaks the data into TF fraction for training "
                    "(default=0.75)")
parser.add_argument("--valid-frac", type=float,
                    metavar="VF", default=0.05,
                    help="breaks the data into VF fraction for validation "
                    "(default=0.05)")
parser.add_argument('--seed', type=int,
                    metavar='S', default=1,
                    help='random seed (default=1)')
parser.add_argument("--locations", type=int,
                    metavar="J", default=116,
                    help="inputs the number of locations (default=116)")
parser.add_argument("--time", type=int,
                    metavar="T", default=182,
                    help="inputs total time of data collection; number of "
                    "weeks (default=182)")
parser.add_argument("--rand",
                    action="store_true", default=False,
                    help="uses random data")

# plot/log options
parser.add_argument("--no-plots", action="store_true", default=False,
                    help="skips generating plot maps")
parser.add_argument("--show-loss-plot", action="store_true", default=False,
                    help="shows the loss plot")
parser.add_argument("--show-map-plot", action="store_true", default=False,
                    help="shows the map plot")

# deprecated options -- not deleting if one chooses to use them
# if using SGD. Remember to check the model's optimizer in this file.
parser.add_argument("--momentum", type=float,
                    metavar="M", default=1.0,
                    help="DEPRECATED: [see script] inputs SGD momentum "
                    "(default=1.0)")

args = parser.parse_args()

if args.layers < 2:
    print("Not possible to construct a network with less than 2 layers")
    exit()

# assigning cuda check and test check to single variables
args.cuda = not args.no_cuda and torch.cuda.is_available()

# set the seeds
torch.manual_seed(args.seed)
np.random.seed(seed=args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# =============================================================================
# constants and parameters
# =============================================================================

# global values and datasets
N = args.layers
N_WEIGHTS = N - 1   # no. of weights is 1 less than no. of layers
J, T, num_features = args.locations, args.time, 0
F_DIST, is_avi = [], []
data_set = {
    'train': {'X': [], 'Y': [], 'R': [], 'u': []},
    'valid': {'X': [], 'Y': [], 'R': [], 'u': []},
    'test': {'X': [], 'Y': [], 'R': [], 'u': []}
}
eta_matrix = []

NUM_VALID = max(int(np.around(args.valid_frac * T)) - 1, 0)
NUM_TRAIN = max(int(np.around(args.train_frac * T)) - 1, 0)
NUM_TEST = T - NUM_TRAIN - NUM_VALID
if NUM_TEST < 0:
    raise RuntimeError("Fractions for training and validation data split add "
                       "up to more than 1")
elif NUM_TEST == 0:
    print("Warning: All data being used for either training or validation, none "
          "for testing")

VALID_ENABLED = NUM_VALID != 0

ORIG_DATA_FILES, RAND_DATA_FILES = ad.read_data_settings_file(
    args.data_settings_file
)

# misc settings and constants
matplotlib.rcParams.update({'font.size': 14})   # font-size for plots
WEIGHT_SAVE_FMT = "%.10f"

# =============================================================================
# forward network arch
# =============================================================================
def batch_norm_tensor(t):
    """
    Normalize tensor t to 0 mean and 1 variance.
    """
    return (t - t.mean()) / torch.sqrt(t.var())

def forward_step_layer(t1, t2, activation_f=torchfun.relu):
    """
    Computes one step of input tensor t1 acted on tensor t2, emulating one layer
    multiplication in the forward step.

    Steps:
        - t1 is batch-multiplied with t2.
        - activation_f is applied to the result.
        - The result is normalized to 0 mean and 1 variance.

    Args:
        t1 -- (torch.Tensor) Tensor 1
        t2 -- (torch.Tensor) Tensor 2
        activation_f -- (fun) Activation function
            (default=torch.nn.functional.relu)
    """
    return batch_norm_tensor(activation_f(t1.bmm(t2)))

def feed_forward(w_list, inp):
    """
    Completes the forward computation for the model. inp is reductively
    multiplied by weight tensors in w_list in order, after which the
    softmax is returned.

    Args:
        w_list -- (list of torch.Tensor or a nn.ParameterList) iterable
            collection of weights multiplied in order
        inp -- (torch.Tensor) input with which to multiply

    Returns:
        torch.Tensor -- a tensor that is the transpose of P
    """
    # last w doesn't need relu
    inp = reduce(forward_step_layer, w_list[:-1], inp)
    inp = inp.bmm(w_list[-1]).view(-1, J)
    inp += eta_matrix

    return torchfun.softmax(inp, dim=1)

# =============================================================================
# data input functions
# =============================================================================

def read_FDIST():
    """
    Reads F and DIST data from the files, and combines them to make F_DIST,
    which gets weighed in the network. Helper to read_data_set().
    """
    global F_DIST, num_features, is_avi

    F, is_avi = ad.read_F_file(
        RAND_DATA_FILES['F'] if args.rand else ORIG_DATA_FILES['F'], J
    )
    DIST = ad.read_dist_file(
        RAND_DATA_FILES['DIST'] if args.rand else ORIG_DATA_FILES['DIST'], J
    )

    # process data for the network
    F = ad.normalize(F, along_dim=0, using_max=True)
    DIST = ad.normalize(DIST, using_max=True)

    num_features = len(F[0]) + 1    # extra 1 for the distance element in F_DIST
    F_DIST = torch.from_numpy(ad.combine_DIST_F(F, DIST, J, num_features))
    num_features += 1   # extra 1 for reward, that is added later

def read_XYR():
    """
    Reads X, Y, R data from the files, and splits them into train and test sets
    as pytorch tensors, after reordering them etc. Helper to read_data_set().
    """
    global data_set

    tmp = {'X': [], 'Y': [], 'R': [], 'u': []}
    if args.rand:
        if not os.path.isfile(RAND_DATA_FILES['XYR']):
            # file doesn't exist, make random data, write to file
            X, Y, R = make_rand_data()
            ad.save_rand_XYR(RAND_DATA_FILES['XYR'], X, Y, R, J, T)

    tmp['X'], tmp['Y'], tmp['R'] = ad.read_XYR_file(
        RAND_DATA_FILES['XYR'] if args.rand else ORIG_DATA_FILES['XYR'], J, T
    )

    # u weights for differently weighing locations when calculating losses
    tmp['u'] = np.sum(tmp['Y'], axis=1)

    # normalize X, Y, R using sum along rows
    for k in ('X', 'Y'):    # R not normalized
        tmp[k] = ad.normalize(tmp[k], along_dim=1, using_max=False)

    # split XYR data into train, validation, and test sets
    # shuffle the data
    shuffle_order = np.random.permutation(T)
    tmp = {k: v[shuffle_order] for k, v in tmp.items()}

    # split the data
    for k, mat in tmp.items():
        data_set['train'][k], data_set['valid'][k], data_set['test'][k] = np.split(
            mat, [NUM_TRAIN, NUM_TRAIN+NUM_VALID], axis=0
        )
        for data_t in data_set:
            # change these np arrays to pytorch tensors
            data_set[data_t][k] = torch.from_numpy(data_set[data_t][k])

def read_data_set():
    """
    Reads Datasets X, Y, R, F, D from the files using avicaching_data
    module's functions. F and D are then combined into F_DIST as preprocessed
    tensor. All datasets are normalized, expanded, averaged as required,
    leaving as torch tensors at the end of the function.
    """
    # shapes of datasets
    # - X, Y: T x J
    # - R: T x J
    # all but last weight tensor: J x num_features x num_features
    # last weight tensor: J x num_features x 1
    # - F_DIST: J x J x num_features

    read_FDIST()
    read_XYR()

# =============================================================================
# data output functions
# =============================================================================

def save_weights(fname, w):
    """
    Saves the weights in w to file fname.

    Args:
        fname -- (str) name of the weights file (without the extension)
        w -- (list of NumPy ndarray) list of weights to be saved; written
            to the file in the order of list
    """
    with open(fname + ".txt", "w") as f:
        # save wi
        f.write('{{"LAYERS": {0:d}, "J": {1:d}, '
                '"NUM_FEATURES": {2:d}}}\n'.format(N, J, num_features))
        for i, wi in enumerate(map(lambda wi: wi.data.cpu().numpy(), w)):
            num_dim = len(list(wi.shape))
            if num_dim > 3:
                print("Not able to write w{0:d} because the no. of dim ({1:d}) "
                      "exceeds 3".format(i+1, num_dim))
            elif num_dim == 3:
                # can be written slice by slice
                f.write("# w{0:d} shape: {1}\n".format(i+1, wi.shape))
                for data_slice in wi:
                    f.write('# New slice\n')
                    np.savetxt(f, data_slice, fmt=WEIGHT_SAVE_FMT, delimiter=" ")
            else:
                # can be written directly
                f.write('# w{0:d} shape: {1}\n'.format(i+1, wi.shape))
                np.savetxt(f, wi, fmt=WEIGHT_SAVE_FMT, delimiter=" ")

def make_rand_data(X_max=100.0, R_max=100.0):
    """
    Creates random X and R and calculates Y based on random weights. Also
    stores the weights in files before returning.

    Args:
        X_max -- (float) Maximum value of element in X dataset (default=100.0)
        R_max -- (float) Maximum value of element in R dataset (default=100.0)

    Returns:
        3-tuple -- (X, Y, R) (values are normalized)
    """
    global F_DIST, eta_matrix

    # create random X and R and w
    origX = np.floor(np.random.rand(T, J) * X_max).astype(np.float32)
    origR = np.floor(np.random.rand(T, J) * R_max).astype(np.float32)

    # all but last set of weights
    w = [torch.randn(J, num_features, num_features) for _ in range(N_WEIGHTS-1)]
    w.append(torch.randn(J, num_features, 1))

    # convert to pytorch tensors, create placeholder for Y
    X = torch.from_numpy(ad.normalize(origX, along_dim=1, using_max=False))
    Y = torch.empty(T, J)
    R = torch.from_numpy(ad.normalize(origR, along_dim=0, using_max=False))
    eta_matrix = args.eta * torch.eye(J)

    if args.cuda:
        # transfer to GPU
        X, Y, R, F_DIST = X.cuda(), Y.cuda(), R.cuda(), F_DIST.cuda()
        w = list(map(lambda wi: wi.cuda(), w))
        eta_matrix = eta_matrix.cuda()

    # build Y
    for t in range(T):
        # build the input by appending testR[t]
        inp = build_input(R[t])
        if args.cuda:
            inp = inp.cuda()

        P = feed_forward(w, inp).t()

        # calculate Y
        Y[t] = P.mv(X[t])

    w[-1].data = w[-1].data.view(-1, num_features)  # remove the extra dim in last wi
    save_weights(os.path.splitext(RAND_DATA_FILES['XYR_weights'])[0], w) # for later verification
    return (X.data.cpu().numpy(), Y.data.cpu().numpy(), R.data.cpu().numpy())

def test_given_data(xyru_data, w, J, T):
    """
    Deprecated, loss value calculation techniques have changed.

    Tests a given set of datasets, printing the loss value after one
    forward propagation.

    Args:
        xyru_data -- (dict) keys 'X', 'Y', 'R', and 'u' must be present
            for testing the weights
        All arguments are self-explanatory
        w -- (list of torch.Tensor) ordered weight tensors
    """
    warnings.warn(
        "test_given_data is deprecated: loss value calculation techniques are "
        "not up to date.",
        DeprecationWarning
    )
    # loss_normalizer divides the calculated loss after feed forward
    # formula = || ((u * (Y-mean(Y)))^2 ||
    Y = xyru_data['Y']
    # TODO: fix this normalizer
    loss_normalizer = (Y - Y.mean()).t().mv(xyru_data['u']).norm(2) ** 2
    loss = 0

    for t in range(T):
        # build the input by appending testR[t]
        inp = build_input(xyru_data['R'][t])
        if args.cuda:
            inp = inp.cuda()

        P = feed_forward(w, inp).t()

        # calculate loss
        Pxt = P.mv(xyru_data['X'][t])
        loss += (xyru_data['u'][t] * (Y[t] - Pxt)).norm(2) ** 2
    # loss += (args.lambda_L2 * reduce(lambda acc, wi: acc + wi.norm(2), w, 0))
    loss /= loss_normalizer
    print("Loss = {0:f}\n".format(loss), end="")

# =============================================================================
# IdProbNet class
# =============================================================================

class IdProbNet(nn.Module):
    """
    An instance of this class emulates the model used for Identification
    Problem as a N-layered network.
    """

    def __init__(self):
        """Initializes IdProbNet, creates the sets of weights for the model."""
        super(IdProbNet, self).__init__()

        self.w = [nn.Parameter(torch.empty(J, num_features, num_features)) \
                  for _ in range(N_WEIGHTS-1)]    # all but last set of weights
        self.w.append(nn.Parameter(torch.empty(J, num_features, 1)))

        # initialize using a distribution
        self.w = nn.ParameterList(map(nn.init.normal_, self.w))

    def forward(self, inp):
        """
        Goes forward in the network -- multiply the weights, apply relu,
        multiply weights again and apply softmax

        Returns:
            torch.Tensor -- result after going forward in the network.
        """
        return feed_forward(self.w, inp)

# =============================================================================
# training and testing routines
# =============================================================================

def train(net, optimizer, loss_normalizer):
    """
    Trains the Neural Network using IdProbNet on the training set.

    Args:
        net -- (IdProbNet instance)
        optimizer -- (torch.optim instance) of the Gradient-Descent function
        loss_normalizer -- (Torch.Tensor) value to be divided from the loss

    Returns:
        3-tuple -- (Execution Time, End loss value,
            Model's prediction after feed forward [Px])
    """
    loss, loss_data, loop_time = 0, 0, 0
    P_data = torch.zeros(NUM_TRAIN, J)

    for t in range(NUM_TRAIN):
        # build the input by appending train R[t] to F_DIST
        inp = build_input(data_set['train']['R'][t])

        loop_start = time.time()
        if args.cuda:
            inp = inp.cuda()

        # feed in data
        P = net(inp).t()    # P is now weighted -> softmax

        # calculate loss
        Pxt = P.mv(data_set['train']['X'][t])
        P_data[t] = Pxt.data
        loss += (data_set['train']['u'][t] * \
                 (data_set['train']['Y'][t] - Pxt)).norm(2) ** 2

        loop_time += (time.time() - loop_start)

    loss_data = float(loss.data)  # copy loss value before adding regularizer
    start_outside = time.time()
    # add regularizer to loss
    loss += (args.lambda_L2 * reduce(lambda acc, wi: acc + wi.norm(2), net.w, 0))

    # backpropagate
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    end_time = (time.time() - start_outside) + loop_time
    loss_data /= loss_normalizer
    return (end_time, loss_data, P_data.mean(dim=0).cpu().numpy())

def run_model_data_t(net, loss_normalizer, NUM_T, data_t):
    """
    Runs the Neural Network using IdProbNet on data_set[data_t].

    Args:
        net -- (IdProbNet instance)
        loss_normalizer -- (Torch.Tensor) value to be divided from the loss
        NUM_T -- (int) number of data instances in data_set[data_t][k] for all k
        data_t -- (str) key in data_set specifying the type of data set to run
            the model on

    Returns:
        3-tuple -- (Execution Time, End loss value,
            Model's prediction after feed forward [Px])
    """
    loss, loop_time = 0, 0
    P_data = torch.zeros(NUM_T, J)

    for t in range(NUM_T):
        # build the input by appending data_set[data_t] R[t]
        inp = build_input(data_set[data_t]['R'][t])

        loop_start = time.time()
        if args.cuda:
            inp = inp.cuda()

        # feed in data
        P = net(inp).t()    # P is now weighted -> softmax

        # calculate loss
        Pxt = P.mv(data_set[data_t]['X'][t])
        P_data[t] = Pxt.data
        loss += (data_set[data_t]['u'][t] * \
                 (data_set[data_t]['Y'][t] - Pxt)).norm(2) ** 2

        loop_time += (time.time() - loop_start)

    start_outside = time.time()
    loss /= loss_normalizer

    end_time = (time.time() - start_outside) + loop_time

    # network's prediction is P_data operated per location, i.e., along
    # time units T (for each P_data column)
    return (end_time, loss.data, P_data.mean(dim=0).cpu().numpy())

def valid(net, loss_normalizer):
    """
    Validates the Neural Network using IdProbNet on the validation set.

    Args:
        net -- (IdProbNet instance)
        loss_normalizer -- (Torch.Tensor) value to be divided from the loss

    Returns:
        3-tuple -- (Execution Time, End loss value,
            Model's prediction after feed forward [Px])
    """
    return run_model_data_t(net, loss_normalizer, NUM_VALID, 'valid')

def test(net, loss_normalizer):
    """
    Tests the Neural Network using IdProbNet on the test set.

    Args:
        net -- (IdProbNet instance)
        loss_normalizer -- (Torch.Tensor) value to be divided from the loss

    Returns:
        3-tuple -- (Execution Time, End loss value,
            Model's prediction after feed forward [Px])
    """
    return run_model_data_t(net, loss_normalizer, NUM_TEST, 'test')

# =============================================================================
# utility functions for training and testing routines
# =============================================================================
def build_input(rt):
    """
    Builds and returns the input for the neural network. Joins F_DIST and R,
    expanding R to fit the dimension.

    Args:
        rt -- (Torch.Tensor) rewards vector to be appended to form the full
            dataset

    Returns:
        Torch.Tensor -- Input dataset for the neural network
    """
    # supplied rt is a vector, repeat on the first dim
    # then append the matrix to the back of F_DIST
    return torch.cat([F_DIST, rt.repeat(J, 1).unsqueeze(2)], dim=2)


def calc_norm_MSE_loss(Y_pred, Y, u):
    """
    Calculates the normalized MSE loss using ground truth Y, visit density
    predictions Y_pred, and loss weights u.

    Formula for normalized MSE loss:
        Fro-norm( diag(u)(Y - Y_pred) )^2 / Fro-norm( diag(u)(Y - mean(Y)) )^2

    The formula aligns with the Identification Problem's metric used in the
    study. mean(Y) is row-wise mean of Y.

    Args:
        Y_pred -- (torch.Tensor) T by J tensor of visit density predictions.
        Y -- (torch.Tensor) T by J tensor of ground truth future density
            predictions.
        u -- (torch.Tensor) T-length vector of weights for calculating loss.
    """
    # Normalizer is diag(u)(Y - mean(Y)), obtained by making 1d data['u']
    # 2d and multiplying element-wise
    normalizer = u.unsqueeze(-1) * (Y - Y.mean(1, keepdim=True))

    # Loss due to predictions X = Y
    loss_preds = u.unsqueeze(-1) * (Y - Y_pred)

    # Calculate Frobenius norm of both losses and divide to get normalized
    # MSE loss
    return (loss_preds.norm(2) ** 2) / (normalizer.norm(2) ** 2)


# =============================================================================
# logs and plots
# =============================================================================

def save_plot(fname, x, y, xlabel, ylabel, title):
    """
    Saves and (optionally) shows the loss plot of train and test periods.

    Args:
        fname -- (str) name of the file for saving (without the extension)
        x -- (NumPy ndarray) data on the x-axis
        y -- (dict of 2d array) data on the y-axis. y should contain keys
            data_t 'train', 'valid', and 'test' which should map to results
            as such: y[data_t][k] should be the results after the k+1 epoch
            such that y[data_t][k][0] is the execution time and y[data_t][k][1]
            is the end loss. See the main area of the script on how this is built.
        xlabel -- (str) label for the x-axis
        ylabel -- (str) what else can it mean?
        title -- (str) title of the plot
    """
    # get the loss values from data
    losses = {data_t: [i for j in v for i in j][1::2] for data_t, v in y.items()}

    # plot details
    handles = []
    loss_fig = plt.figure(1)
    train_label, = plt.plot(x, losses['train'], "r-", label="Train Loss")
    handles.append(train_label)
    if VALID_ENABLED:
        valid_label, = plt.plot(x, losses['valid'], "b-", label="Validation Loss")
        handles.append(valid_label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, which="major", axis="both",
             color="k", ls="dotted", lw="1.0")
    plt.grid(True, which="minor", axis="y", color="k", ls="dotted", lw="0.5")
    plt.minorticks_on()
    plt.title(title)

    # check if testing was enabled
    if args.record_test_res:
        test_label, = plt.plot(x, losses['test'], "g-", label="Test Loss")
        handles.append(test_label)
    plt.legend(handles=handles)

    # save and show
    loss_fig.tight_layout()
    loss_fig.savefig(fname + ".png", bbox_inches="tight", dpi=200)
    if args.show_loss_plot:
        plt.show()
    plt.close()

def save_log(fname, x, y, t_time):
    """
    Saves the log of train and test periods to a file.

    Args:
        fname -- (str) name of the file (without the extension)
        x -- (NumPy ndarray) epoch data [1..number_of_epochs]
        y -- (dict of 2d array) same as that of save_plot()
        t_time -- (flaot) runtime of the model
    """

    # Calculate baseline normalized MSE loss for train, test, and valid datasets
    # using historical and random predictions X as Y
    baseline_preds = {'rand': {}, 'hist': {}}
    for data_t, data in data_set.items():
        if (data_t == 'valid' and VALID_ENABLED) or data_t != 'valid':
            rand_y = torchfun.normalize(torch.rand(data['Y'].shape), p=1, dim=1)
            if args.cuda:
                rand_y = rand_y.cuda()
            baseline_preds['rand'][data_t] = calc_norm_MSE_loss(rand_y, data['Y'], data['u'])
            baseline_preds['hist'][data_t] = calc_norm_MSE_loss(data['X'], data['Y'], data['u'])

    # Calculate baseline normalized MSE loss for historical preds for the full dataset combined
    total_dataset = combine_datasets()
    baseline_preds['hist']['total'] = calc_norm_MSE_loss(
        total_dataset['X'], total_dataset['Y'], total_dataset['u']
    )

    # write summary of logs, and the parameters used
    with open(fname + ".json", "wt") as f:
        len_x = len(x)
        j = {
            # data params
            'J': J,
            'T': T,
            'num_features': num_features,
            'train_frac': args.train_frac,
            'valid_frac': args.valid_frac,
            # model hyperparams
            'epochs': len_x,
            'layers': N,
            'eta': args.eta,
            'lambda_L2': args.lambda_L2,
            # misc params
            'seed': args.seed,
            # end results
            'runtime': t_time,
            'end_train_acc': float(y['train'][len_x-1][1]),
        }

        for baseline_t, losses_dict in baseline_preds.items():
            for data_t, loss in losses_dict.items():
                j['baseline_{:s}_loss_{:s}'.format(baseline_t, data_t)] = float(loss)

        if VALID_ENABLED:
            j['end_valid_acc'] = float(y['valid'][len_x-1][1])

        if args.record_test_res:
            j['end_test_acc'] = float(y['test'][len_x-1][1])

        if args.rand:
            j['device'] = 'gpu' if args.cuda else 'cpu'

        json.dump(j, f, indent=4, separators=(',', ': '))

    # write detailed logs
    with open(fname + ".csv", "wt") as f:
        # write header
        f.write("epoch,train_loss,train_time,valid_loss,valid_time")
        if args.record_test_res:
            f.write(",test_loss,test_time")
        f.write("\n")

        for i, xi in enumerate(x):
            f.write("{e:d},{t_l:0.4f},{t_t:0.4f},{v_l:0.4f},{v_t:0.4f}".format(
                e=xi, t_l=y['train'][i][1], t_t=y['train'][i][0],
                v_l=y['valid'][i][1], v_t=y['valid'][i][0]
            ))
            if args.record_test_res:
                f.write(",{t_l:0.4f},{t_t:0.4f}".format(
                    t_l=y['test'][i][1], t_t=y['test'][i][0]
                ))
            f.write("\n")

def find_idx_of_nearest_el(array, value):
    """
    Helper function to plot_predicted_map(). Returns the index of the element in
    array closest to value

    Args:
        array -- (NumPy ndarray) array to be searched in
        value -- (float) closest number in array found for this number

    Returns:
        int -- index of the closest number to value in array
    """
    return (np.abs(array - value)).argmin()


def plot_predicted_map(fname, lat_long, point_info, title, plot_offset=0.05):
    """
    Plots the a scatter plot of point_info on the map specified by the latitudes
    and longitudes and saves the plot to a image file

    Args:
        fname -- (str) file name of the plot (without the extension)
        lat_long -- (NumPy ndarray) 2-d matrix of latitudes and longitudes of
            locations. The first column contains latitudes, and the second
            column contains longitudes.
        point_info -- (NumPy ndarray) Z values for all locations. The order of
            locations must be same as the order in lat_long
        title -- (str) title of the plot
        plot_offset -- (float) padding value for latitude and longitude in the
            plot (default=0.05)
    """
    # extract latitude and longitude
    lati = lat_long[:, 0]
    longi = lat_long[:, 1]
    # calculate plot dimensions - select between latitude/longitude based on
    # their span over earth. The greater span is the basis
    lo_min, lo_max = min(longi) - plot_offset, max(longi) + plot_offset
    la_min, la_max = min(lati) - plot_offset, max(lati) + plot_offset
    plot_width = max(lo_max - lo_min, la_max - la_min)
    lo_max = lo_min + plot_width
    la_max = la_min + plot_width

    # create the mesh for pcolormesh, see its documentation
    # retained step for convenience in testing
    # J+10 values needed on each side, this can lead to rectangular dots
    lo_range = np.linspace(lo_min, lo_max, num=J+10, retstep=True)
    la_range = np.linspace(la_min, la_max, num=J+10, retstep=True)
    lo, la = np.meshgrid(lo_range[0], la_range[0])

    z = np.zeros([J + 10, J + 10])
    for k in range(J):
        # for each location in latitude and longitude array, find the closest
        # value in the mesh, i.e., lati[k] in the mesh, longi[k] in the mesh
        lo_k_mesh = find_idx_of_nearest_el(lo[0], longi[k])
        la_k_mesh = find_idx_of_nearest_el(la[:, 0], lati[k])
        z[lo_k_mesh][la_k_mesh] = point_info[k]  # assign Z value in the matrix

    map_fig = plt.figure(2)
    plt.pcolormesh(lo, la, z, cmap=plt.cm.get_cmap('Greys'), vmin=0.0, vmax=0.01)
    plt.axis([lo.min(), lo.max(), la.min(), la.max()])
    plt.colorbar()
    plt.title(title)

    map_fig.tight_layout()
    map_fig.savefig(fname + ".png", bbox_inches="tight", dpi=200)
    if args.show_map_plot:
        plt.show()
    plt.close()

# =============================================================================
# misc utility functions
# =============================================================================
def combine_datasets():
    """
    Returns a dictionary d with keys k same as keys of data_set[data_t]
    for any data_t, such that d[k] is a concatenated Tensor of
    data_set[data_t][k] for all data_t.
    """
    return {
        k: torch.cat([data[k] for _, data in data_set.items()], dim=0)
        for k in data_set['train'].keys()   # keys collected from any data_t
    }


# =============================================================================
# main program
# =============================================================================
if __name__ == "__main__":
    # READY!!
    read_data_set()
    net = IdProbNet()

    # SET!!
    eta_matrix = args.eta * torch.eye(J)
    transfer_time = time.time()
    if args.cuda:
        # transfer tensors to the gpu
        net.cuda()
        for data_t in data_set:
            data_set[data_t] = {k: v.cuda() for k, v in data_set[data_t].items()}
        F_DIST = F_DIST.cuda()
        eta_matrix = eta_matrix.cuda()
        file_pre_gpu = "gpu, "
    else:
        file_pre_gpu = "cpu, "
    transfer_time = time.time() - transfer_time

    # optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)
    optimizer = optim.Adam(net.parameters(), lr=args.lr)

    # formula = sum_t (u_t(Y_t-mean(Y_t)))^2
    loss_normalizer = {}
    for data_t, data in data_set.items():
        if data['Y'].shape[0] == 0:
            # Empty tensor, nothing to normalize by
            loss_normalizer[data_t] = torch.Tensor([1.])
        else:
            # See calc_norm_MSE_loss for this calculation
            loss_normalizer[data_t] = (
                data['u'].unsqueeze(-1) * \
                (data['Y'] - data['Y'].mean(1, keepdim=True))
            ).norm(2) ** 2

    # GO!!
    time_loss = {'train': [], 'valid': [], 'test': []}
    total_time = transfer_time
    y_pred = np.empty(J)

    def run_function(data_t, start_print="", end_print=""):
        """
        Runs the function associated with data_t.

        Args:
            data_t -- (str) either 'train', 'valid', or 'test'
            start_print -- (str) text to print before loss (default="")
            end_print -- (str) text to print after loss (default="")

        Returns:
            NumPy ndarray -- prediction of running function associated with
                data_t
        """
        global total_time, time_loss
        f_args = [net, optimizer] if data_t == 'train' else [net]
        f_args.append(loss_normalizer[data_t])

        # call the function associated with data_t with f_args
        res = globals()[data_t](*f_args)    # expands f_args list into sep vars

        # the third element is not logged
        time_loss[data_t].append(res[0:2])
        total_time += res[0]

        # print results, some quirky arguments to print for nice console printing
        if e % 20 == 0:
            print("{0:s}, {1:s}loss={2:0.8f}{3:s}".format(
                start_print, data_t, res[1], end_print
            ), end="")

        return res[2]

    for e in range(args.epochs):
        train_res = run_function('train', start_print="e={0:5d}".format(e),
                    end_print=("" if NUM_VALID or args.record_test_res else "\n"))

        if NUM_VALID:
            valid_res = run_function('valid',
                                     end_print=("" if args.record_test_res else "\n"))
        else:
            time_loss['valid'].append([0, 0])

        if args.record_test_res:
            test_res = run_function('test', end_print="\n")

        # Save network's final prediction y_pred if last epoch
        if e == args.epochs-1:
            if args.record_test_res:
                y_pred = test_res
            elif NUM_VALID:
                y_pred = valid_res
            else:
                y_pred = train_res

    total_epochs = args.epochs

    # FINISH!!
    # log and plot the results: epoch vs loss

    # define file names
    file_pre = "{0:s}XYR_seed={1:d}, epochs={2:d}, ".format(
        "rand" if args.rand else "orig", args.seed, total_epochs
    )
    lat_long = ad.read_lat_long_from_Ffile(
        RAND_DATA_FILES['F'] if args.rand else ORIG_DATA_FILES['F'], J
    )
    log_name = "train={0:3.0f}%, valid={1:3.0f}%, lr={2:0.3e}, time={3:0.4f} sec".format(
        args.train_frac*100, args.valid_frac*100, args.lr, total_time
    )
    epoch_data = np.arange(total_epochs)
    fname = "{0:s}{1:d}layer_".format(
        "test_on, " if args.record_test_res else "", N
    ) + file_pre_gpu + file_pre + log_name

    # save amd plot data
    save_log(
        "./stats/find_weights/logs/" + fname, epoch_data, time_loss, total_time
    )

    net.w[-1].data = net.w[-1].data.view(-1, num_features)  # remove the extra dim in last wi
    save_weights("./stats/find_weights/weights/" + fname, net.w)

    if not args.no_plots:
        # should plot
        save_plot(
            "./stats/find_weights/plots/" + fname, epoch_data,
            time_loss, "epoch", "loss", log_name
        )
        plot_predicted_map(
            "./stats/find_weights/map_plots/" + fname,
            lat_long, y_pred, log_name
        )

    print("---> " + fname + " DONE")
