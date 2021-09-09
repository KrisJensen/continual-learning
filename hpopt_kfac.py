#!/usr/bin/env python3
import argparse
import os
import numpy as np
from param_stamp import get_param_stamp_from_args
import visual_plt
import main
import utils
import pickle
from param_values import set_default_values


description = 'Compare CL strategies using various metrics on each scenario of permuted or split MNIST.'
parser = argparse.ArgumentParser('./hpopt_kfac.py', description=description)
parser.add_argument('--seed', type=int, default=30, help='[first] random seed (for each random-module used)')
parser.add_argument('--n-seeds', type=int, default=15, help='how often to repeat?')
parser.add_argument('--no-gpus', action='store_false', dest='cuda', help="don't use GPUs")
parser.add_argument('--data-dir', type=str, default='./datasets', dest='d_dir', help="default: %(default)s")
parser.add_argument('--plot-dir', type=str, default='./plots', dest='p_dir', help="default: %(default)s")
parser.add_argument('--results-dir', type=str, default='./results', dest='r_dir', help="default: %(default)s")

# expirimental task parameters.
task_params = parser.add_argument_group('Task Parameters')
task_params.add_argument('--experiment', type=str, default='splitMNIST', choices=['permMNIST', 'splitMNIST'])
task_params.add_argument('--scenario', type=str, default='task', choices=['task', 'domain', 'class'])
task_params.add_argument('--tasks', type=int, help='number of tasks')

# specify loss functions to be used
loss_params = parser.add_argument_group('Loss Parameters')
loss_params.add_argument('--bce', action='store_true', help="use binary (instead of multi-class) classication loss")

# model architecture parameters
model_params = parser.add_argument_group('Parameters Main Model')
model_params.add_argument('--fc-layers', type=int, default=3, dest='fc_lay', help="# of fully-connected layers")
model_params.add_argument('--fc-units', type=int, metavar="N", help="# of units in first fc-layers")
model_params.add_argument('--fc-drop', type=float, default=0., help="dropout probability for fc-units")
model_params.add_argument('--fc-bn', type=str, default="no", help="use batch-norm in the fc-layers (no|yes)")
model_params.add_argument('--fc-nl', type=str, default="relu", choices=["relu", "leakyrelu"])
model_params.add_argument('--singlehead', action='store_true', help="for Task-IL: use a 'single-headed' output layer   "
                                                                   " (instead of a 'multi-headed' one)")

# training hyperparameters / initialization
train_params = parser.add_argument_group('Training Parameters')
train_params.add_argument('--iters', type=int, help="# batches to optimize solver")
train_params.add_argument('--lr', type=float, help="learning rate")
train_params.add_argument('--batch', type=int, default=256, help="batch-size")
train_params.add_argument('--optimizer', type=str, choices=['adam', 'adam_reset', 'sgd'], default='adam')

# "memory replay" parameters
replay_params = parser.add_argument_group('Replay Parameters')
replay_params.add_argument('--temp', type=float, default=2., dest='temp', help="temperature for distillation")
# -generative model parameters (if separate model)
genmodel_params = parser.add_argument_group('Generative Model Parameters')
genmodel_params.add_argument('--g-z-dim', type=int, default=100, help='size of latent representation (default: 100)')
genmodel_params.add_argument('--g-fc-lay', type=int, help='[fc_layers] in generator (default: same as classifier)')
genmodel_params.add_argument('--g-fc-uni', type=int, help='[fc_units] in generator (default: same as classifier)')
# - hyper-parameters for generative model (if separate model)
gen_params = parser.add_argument_group('Generator Hyper Parameters')
gen_params.add_argument('--g-iters', type=int, help="# batches to train generator (default: as classifier)")
gen_params.add_argument('--lr-gen', type=float, help="learning rate generator (default: lr)")

# "memory allocation" parameters
cl_params = parser.add_argument_group('Memory Allocation Parameters')
cl_params.add_argument('--lambda', type=float, dest="ewc_lambda", help="--> EWC: regularisation strength")
cl_params.add_argument('--o-lambda', type=float, help="--> online EWC: regularisation strength")
cl_params.add_argument('--fisher-n', type=int, help="--> EWC: sample size estimating Fisher Information")
cl_params.add_argument('--gamma', type=float, help="--> EWC: forgetting coefficient (for 'online EWC')")
cl_params.add_argument('--emp-fi', action='store_true', help="--> EWC: estimate FI with provided labels")
cl_params.add_argument('--c', type=float, dest="si_c", help="--> SI: regularisation strength")
cl_params.add_argument('--epsilon', type=float, default=0.1, dest="epsilon", help="--> SI: dampening parameter")
cl_params.add_argument('--gating-prop', type=float, metavar="PROP", help="--> XdG: prop neurons per layer to gate")
cl_params.add_argument('--online', action='store_true', help="--> EWC: perform 'online EWC'")

# iCaRL parameters
icarl_params = parser.add_argument_group('iCaRL Parameters')
icarl_params.add_argument('--budget', type=int, default=1000, dest="budget", help="how many exemplars can be stored?")
icarl_params.add_argument('--herding', action='store_true', help="use herding to select exemplars (instead of random)")
icarl_params.add_argument('--use-exemplars', action='store_true', help="use stored exemplars for classification?")
icarl_params.add_argument('--norm-exemplars', action='store_true', help="normalize features/averages of exemplars")

# evaluation parameters
eval_params = parser.add_argument_group('Evaluation Parameters')
eval_params.add_argument('--time', action='store_true', help="keep track of total training time")
eval_params.add_argument('--pdf', action='store_true', help="generate pdfs for individual experiments")
eval_params.add_argument('--visdom', action='store_true', help="use visdom for on-the-fly plots")
eval_params.add_argument('--prec-n', type=int, default=1024, help="# samples for evaluating solver's precision")
eval_params.add_argument('--sample-n', type=int, default=64, help="# images to show")

# NCL parameters
cl_params.add_argument('--ncl', action='store_true', help="use 'NCL' ")
cl_params.add_argument('--kfncl', action='store_true', help="use 'KF NCL' ")
train_params.add_argument('--alpha',
                          type=float,
                          help="regularization alpha")
train_params.add_argument('--data_size',
                          type=float,
                          help="prior data size")
train_params.add_argument('--momentum',
                          type=float,
                          help="momentum to use with SGD")
train_params.add_argument('--cudanum',
                         type = str,
                         default='default',
                         help="which cuda? (e.g. cuda:0)")

## KFAC parameters
cl_params.add_argument('--ewc_kfac', action='store_true', help="use 'EWC with KFAC' (Ritter et al. 2018) ")

## projection parameters ##
cl_params.add_argument('--owm', action='store_true', help="use orthogonal weight modification (Zeng et al. 2018) ")


def get_results(args):
    # -get param-stamp
    param_stamp = get_param_stamp_from_args(args)
    # -check whether already run; if not do so
    if os.path.isfile("{}/dict-{}.pkl".format(args.r_dir, param_stamp)):
        print("{}: already run".format(param_stamp))
    else:
        print("{}: ...running...".format(param_stamp))
        main.run(args)
    # -get results-dict
    dict = utils.load_object("{}/dict-{}".format(args.r_dir, param_stamp))
    # -get average precision
    fileName = '{}/prec-{}.txt'.format(args.r_dir, param_stamp)
    file = open(fileName)
    ave = float(file.readline())
    file.close()
    # -print average precision on screen
    print("--> average precision: {}".format(ave))
    # -return average precision
    return (dict, ave)


def collect_all(method_dict, seed_list, args, name=None):
    # -print name of method on screen
    if name is not None:
        print("\n------{}------".format(name))
    # -run method for all random seeds
    for seed in seed_list:
        args.seed = seed
        method_dict[seed] = get_results(args)
    # -return updated dictionary with results
    return method_dict



if __name__ == '__main__':

    ## Load input-arguments
    args = parser.parse_args()
    # -set default-values for certain arguments based on chosen scenario & experiment
    args = set_default_values(args)
    # -set other default arguments
    args.lr_gen = args.lr if args.lr_gen is None else args.lr_gen
    args.g_iters = args.iters if args.g_iters is None else args.g_iters
    args.g_fc_lay = args.fc_lay if args.g_fc_lay is None else args.g_fc_lay
    args.g_fc_uni = args.fc_units if args.g_fc_uni is None else args.g_fc_uni
    # -create results-directory if needed
    if not os.path.isdir(args.r_dir):
        os.mkdir(args.r_dir)
    # -create plots-directory if needed
    if not os.path.isdir(args.p_dir):
        os.mkdir(args.p_dir)

    ## Add non-optional input argument that will be the same for all runs
    args.metrics = True
    args.feedback = False
    args.log_per_task = True

    ## Add input arguments that will be different for different runs
    args.distill = False
    args.agem = False
    args.ewc = False
    args.online = False
    args.si = False
    args.xdg = False
    args.add_exemplars = False
    args.bce_distill= False
    args.icarl = False
    # args.seed could of course also vary!
    
    if args.owm:
        owm = True
        print('running owm!!')
        args.ewc_kfac = False
        args.optimizer = 'sgd'
        args.lr=5e-2
        if args.experiment == 'splitMNIST':
            lambdas = 10.**np.array([-7, -6,-5,-4,-3,-2,-1,0,1])
        else:
            lambdas = 10.**np.array([-7, -6,-5,-4,-3,-2,-1,0,1])
    else:
        args.ewc_kfac = True
        if args.experiment == 'splitMNIST':
            lambdas = 10.**np.array([0,1,2,3,4,5,6,7,8])
        else:
            lambdas = 10.**np.array([-2,-1,0,1,2,3,4,5,6])

    #-------------------------------------------------------------------------------------------------#

    #--------------------------#
    #----- RUN KFAC MODELS -----#
    #--------------------------#

    seed_list = list(range(args.seed, args.seed+args.n_seeds))
    
    print('\n\nseeds', seed_list)

    ## KFAC
    args.replay = "none"
    args.ewc = False
    args.online = True
    args.gamma = 1.
    
    result = {'lambdas': lambdas}
    
    for ilambda, lambda_ in enumerate(lambdas):
        print('\n\nnew lambda:', lambda_)
        
        if args.owm:
            args.alpha = lambda_
        else:
            args.ewc_lambda = lambda_
            args.o_lambda = lambda_
            
        KFAC = {}
        KFAC = collect_all(KFAC, seed_list, args, name="KFAC")
        result[ilambda] = KFAC
        
    savename = "summary-{}-{}".format(args.experiment, args.scenario)
    
    if args.owm:
        pickle.dump(result, open('ncl_results/hpopt_owm_'+savename+'.p', 'wb'))
    else:
        pickle.dump(result, open('ncl_results/hpopt_kfac_'+savename+'.p', 'wb'))