#!/usr/bin/env python3
import argparse
import os
import numpy as np
from param_stamp import get_param_stamp_from_args
import visual_plt
import main
import pickle
import utils
from param_values import set_default_values


parser = argparse.ArgumentParser('./run_model_repeats.py')

parser.add_argument('--get-stamp',
                    action='store_true',
                    help='print param-stamp & exit')
parser.add_argument('--seed',
                    type=int,
                    default=0,
                    help='random seed (for each random-module used)')
parser.add_argument('--no-gpus',
                    action='store_false',
                    dest='cuda',
                    help="don't use GPUs")
parser.add_argument('--data-dir', type=str, default='./datasets', dest='d_dir', help="default: %(default)s")
parser.add_argument('--plot-dir', type=str, default='./plots', dest='p_dir', help="default: %(default)s")
parser.add_argument('--results-dir', type=str, default='./results', dest='r_dir', help="default: %(default)s")
parser.add_argument('--fname', type=str, default='./ncl_results/test', help="default: %(default)s")
parser.add_argument('--n-seeds', type=int, default=5, help='how often to repeat?')

# expirimental task parameters
task_params = parser.add_argument_group('Task Parameters')
task_params.add_argument('--experiment',
                         type=str,
                         default='splitMNIST',
                         choices=['permMNIST', 'splitMNIST'])
task_params.add_argument('--scenario',
                         type=str,
                         default='class',
                         choices=['task', 'domain', 'class'])
task_params.add_argument('--tasks', type=int, help='number of tasks')

# specify loss functions to be used
loss_params = parser.add_argument_group('Loss Parameters')
loss_params.add_argument(
    '--bce',
    action='store_true',
    help="use binary (instead of multi-class) classication loss")
loss_params.add_argument('--bce-distill',
                         action='store_true',
                         help='distilled loss on previous classes for new'
                         ' examples (only if --bce & --scenario="class")')

# model architecture parameters
model_params = parser.add_argument_group('Model Parameters')
model_params.add_argument('--fc-layers',
                          type=int,
                          default=3,
                          dest='fc_lay',
                          help="# of fully-connected layers")
model_params.add_argument('--fc-units',
                          type=int,
                          metavar="N",
                          help="# of units in first fc-layers")
model_params.add_argument('--fc-drop',
                          type=float,
                          default=0.,
                          help="dropout probability for fc-units")
model_params.add_argument('--fc-bn',
                          type=str,
                          default="no",
                          help="use batch-norm in the fc-layers (no|yes)")
model_params.add_argument('--fc-nl',
                          type=str,
                          default="relu",
                          choices=["relu", "leakyrelu"])
model_params.add_argument(
    '--singlehead',
    action='store_true',
    help="for Task-IL: use a 'single-headed' output layer   "
    " (instead of a 'multi-headed' one)")

# training hyperparameters / initialization
train_params = parser.add_argument_group('Training Parameters')
train_params.add_argument('--iters',
                          type=int,
                          help="# batches to optimize solver")
train_params.add_argument('--lr', type=float, help="learning rate")
train_params.add_argument('--batch', type=int, default=256, help="batch-size")
train_params.add_argument('--optimizer',
                          type=str,
                          choices=['adam', 'adam_reset', 'sgd'],
                          default='adam')

# "memory replay" parameters
replay_params = parser.add_argument_group('Replay Parameters')
replay_params.add_argument('--feedback',
                           action="store_true",
                           help="equip model with feedback connections")
replay_params.add_argument('--z-dim',
                           type=int,
                           default=100,
                           help='size of latent representation (default: 100)')
replay_choices = [
    'offline', 'exact', 'generative', 'none', 'current', 'exemplars'
]
replay_params.add_argument('--replay',
                           type=str,
                           default='none',
                           choices=replay_choices)
replay_params.add_argument('--distill',
                           action='store_true',
                           help="use distillation for replay?")
replay_params.add_argument('--temp',
                           type=float,
                           default=2.,
                           dest='temp',
                           help="temperature for distillation")
replay_params.add_argument(
    '--agem',
    action='store_true',
    help="use gradient of replay as inequality constraint")
# -generative model parameters (if separate model)
genmodel_params = parser.add_argument_group('Generative Model Parameters')
genmodel_params.add_argument(
    '--g-z-dim',
    type=int,
    default=100,
    help='size of latent representation (default: 100)')
genmodel_params.add_argument(
    '--g-fc-lay',
    type=int,
    help='[fc_layers] in generator (default: same as classifier)')
genmodel_params.add_argument(
    '--g-fc-uni',
    type=int,
    help='[fc_units] in generator (default: same as classifier)')
# - hyper-parameters for generative model (if separate model)
gen_params = parser.add_argument_group('Generator Hyper Parameters')
gen_params.add_argument(
    '--g-iters',
    type=int,
    help="# batches to train generator (default: as classifier)")
gen_params.add_argument('--lr-gen',
                        type=float,
                        help="learning rate generator (default: lr)")

# "memory allocation" parameters
cl_params = parser.add_argument_group('Memory Allocation Parameters')
cl_params.add_argument('--ewc',
                       action='store_true',
                       help="use 'EWC' (Kirkpatrick et al, 2017)")
cl_params.add_argument('--lambda',
                       type=float,
                       dest="ewc_lambda",
                       help="--> EWC: regularisation strength")
cl_params.add_argument('--o-lambda', type=float, help="--> online EWC: regularisation strength")
cl_params.add_argument('--kfac-lambda', type=float, help="--> online EWC: KFAC regularisation strength")
cl_params.add_argument(
    '--fisher-n',
    type=int,
    help="--> EWC: sample size estimating Fisher Information")
cl_params.add_argument('--online',
                       action='store_true',
                       help="--> EWC: perform 'online EWC'")
cl_params.add_argument(
    '--gamma',
    type=float,
    default=1.0,
    help="--> EWC: forgetting coefficient (for 'online EWC')")
cl_params.add_argument('--emp-fi',
                       action='store_true',
                       help="--> EWC: estimate FI with provided labels")
cl_params.add_argument(
    '--si',
    action='store_true',
    help="use 'Synaptic Intelligence' (Zenke, Poole et al, 2017)")
cl_params.add_argument('--c',
                       type=float,
                       dest="si_c",
                       help="--> SI: regularisation strength")
cl_params.add_argument('--epsilon',
                       type=float,
                       default=0.1,
                       dest="epsilon",
                       help="--> SI: dampening parameter")
cl_params.add_argument(
    '--xdg',
    action='store_true',
    help="Use 'Context-dependent Gating' (Masse et al, 2018)")
cl_params.add_argument('--gating-prop',
                       type=float,
                       metavar="PROP",
                       help="--> XdG: prop neurons per layer to gate")

# data storage ('exemplars') parameters
store_params = parser.add_argument_group('Data Storage Parameters')
store_params.add_argument('--icarl',
                          action='store_true',
                          help="bce-distill, use-exemplars & add-exemplars")
store_params.add_argument('--use-exemplars',
                          action='store_true',
                          help="use exemplars for classification")
store_params.add_argument('--add-exemplars',
                          action='store_true',
                          help="add exemplars to current task's training set")
store_params.add_argument('--budget',
                          type=int,
                          default=1000,
                          dest="budget",
                          help="how many samples can be stored?")
store_params.add_argument(
    '--herding',
    action='store_true',
    help="use herding to select stored data (instead of random)")
store_params.add_argument('--norm-exemplars',
                          action='store_true',
                          help="normalize features/averages of exemplars")

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
train_params.add_argument('--owm_alpha',
                          type=float,
                          help="regularization alpha for OWM")
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

    ## use appropriate value for lambda
    if args.ewc_kfac:
        args.ewc_lambda = args.kfac_lambda
    elif args.ewc and args.online:
        args.ewc_lambda = args.o_lambda
        
    if args.owm:
        args.alpha = args.owm_alpha

    seed_list = list(range(args.seed, args.seed+args.n_seeds))

    ###----"run task"----####

    result = {}
    result = collect_all(result, seed_list, args, name="result")


    # name for plot
    save_name = args.fname+"_summary-{}-{}".format(args.experiment, args.scenario)
    save_name += '_nseed'+str(len(seed_list))
    
    pickle.dump(result, open(save_name+'.p', 'wb'))

