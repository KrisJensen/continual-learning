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
from pathlib import Path


description = 'Compare CL strategies using various metrics on each scenario of permuted or split MNIST.'
#parser = argparse.ArgumentParser('./compare_all.py', description=description)
parser = argparse.ArgumentParser('./compare_all.py', description=description)
parser.add_argument('--seed', type=int, default=1, help='[first] random seed (for each random-module used)')
parser.add_argument('--n-seeds', type=int, default=1, help='how often to repeat?')
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
                          default=1e-5,
                          help="regularization alpha")
train_params.add_argument('--data_size',
                          type=float,
                          default=12000.,
                          help="prior data size")
train_params.add_argument('--momentum',
                          type=float,
                          default=0.9,
                          help="momentum to use with SGD")
train_params.add_argument('--power',
                          type=int,
                          default=1,
                          help="raise the Fisher to some power?")
train_params.add_argument('--cudanum',
                         type = str,
                         default='default',
                         help="which cuda? (e.g. cuda:0)")

## KFAC parameters
cl_params.add_argument('--ewc_kfac', action='store_true', help="use 'EWC with KFAC' (Ritter et al. 2018) ")

def get_results(args):
    # -get param-stamp
    param_stamp = get_param_stamp_from_args(args)
    # -check whether already run; if not do so
    if os.path.isfile("{}/dict-{}.pkl".format(args.r_dir, param_stamp)):
        print("{}: already run".format(param_stamp))
    else:
        print("{}: ...running...".format(param_stamp))
        main.run(args, verbose=False)
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

    #-------------------------------------------------------------------------------------------------#

    #--------------------------#
    #----- RUN ALL MODELS -----#
    #--------------------------#

    seed_list = list(range(args.seed, args.seed+args.n_seeds))


    ###----"BASELINES"----###

    ## Offline
    args.replay = "offline"
    OFF = {}
    OFF = collect_all(OFF, seed_list, args, name="Offline")

    ## None
    args.replay = "none"
    NONE = {}
    NONE = collect_all(NONE, seed_list, args, name="None")

    ###----"REGULARIZATION"----####
    
    ## NCL ##
    old_lambda = args.ewc_lambda
    old_lr = args.lr
    old_gamma = args.gamma
    args.ncl = True
    args.online = True
    args.ewc_lambda = 1
    args.alpha = 1e-10
    args.gamma = 1.
    args.optimizer = 'sgd'
    args.momentum = 0.9
    args.data_size = 12000
    #args.lr=5e-4
    args.lr=5e-3 if args.experiment == 'splitMNIST' else 5e-2
    NCL = {}
    NCL = collect_all(NCL, seed_list, args, name="NCL")
    args.ncl = False
    
    ## ncl kfac ##
    #args.lr = 5e-4
    #args.lr = 1e-3
    args.kfncl = True
    args.lr=5e-2 if args.experiment == 'splitMNIST' else 5e-2
    KFNCL = {}
    KFNCL = collect_all(KFNCL, seed_list, args, name="KFNCL")
    args.online = False
    args.kfncl = False
    args.optimizer='adam'
    args.lr = old_lr
    args.ewc_lambda = old_lambda
    args.gamma = old_gamma


    ## SI
    args.si = True
    SI = {}
    SI = collect_all(SI, seed_list, args, name="SI")
    args.si = False
    
    ## EWC
    args.ewc = True
    EWC = {}
    EWC = collect_all(EWC, seed_list, args, name="EWC")

    ## online EWC
    args.online = True
    args.ewc_lambda = args.o_lambda
    OEWC = {}
    OEWC = collect_all(OEWC, seed_list, args, name="Online EWC")
    

    #-------------------------------------------------------------------------------------------------#

    #---------------------------#
    #----- COLLECT RESULTS -----#
    #---------------------------#

    ## For each task & seed, get the test accuracy of the "base model"
    key = 'acc per task{}'.format(' (all classes up to evaluated task)' if args.scenario=="class" else '')
    base = {}
    for i in range(args.tasks):
        base["task {}".format(i+1)] = {}
        for seed in seed_list:
            base["task {}".format(i+1)][seed] = OFF[seed][0][key]['task {}'.format(i+1)][i]

    ave_prec = {}
    prec = {}
    ave_BWT = {}
    ave_FWT = {}
    ave_F = {}
    ave_F2 = {}
    ave_I = {}

    ## For each seed, create list with average metrics
    for seed in seed_list:
        ## AVERAGE TEST ACCURACY
        ave_prec[seed] = [NONE[seed][1], OFF[seed][1], EWC[seed][1], OEWC[seed][1], SI[seed][1], NCL[seed][1], KFNCL[seed][1]]
        # -for plot of average accuracy throughout training
        key = "average"
        prec[seed] = [NONE[seed][0][key], OFF[seed][0][key], EWC[seed][0][key], OEWC[seed][0][key], SI[seed][0][key], NCL[seed][0][key], KFNCL[seed][0][key]]

        ## BACKWARD TRANSFER (BWT)
        key = 'BWT'
        ave_BWT[seed] = [NONE[seed][0][key], OFF[seed][0][key], EWC[seed][0][key], OEWC[seed][0][key], SI[seed][0][key], NCL[seed][0][key], KFNCL[seed][0][key]]

        ## FORWARD TRANSFER (FWT)
        key = 'FWT'
        ave_FWT[seed] = [NONE[seed][0][key], OFF[seed][0][key], EWC[seed][0][key], OEWC[seed][0][key], SI[seed][0][key], NCL[seed][0][key], KFNCL[seed][0][key]]

        ## FORGETTING (F)
        key = 'F'
        ave_F[seed] = [NONE[seed][0][key], OFF[seed][0][key], EWC[seed][0][key], OEWC[seed][0][key], SI[seed][0][key], NCL[seed][0][key], KFNCL[seed][0][key]]

        ## INTRANSIGENCE (I)
        key = 'acc per task{}'.format(' (all classes up to evaluated task)' if args.scenario == "class" else '')
        ave_I[seed] = [
            np.mean([(
                OFF[seed][0][key]['task {}'.format(i+1)][i] - NONE[seed][0][key]['task {}'.format(i + 1)][i]
            ) for i in range(1, args.tasks)]),
            np.mean([(
                OFF[seed][0][key]['task {}'.format(i + 1)][i] - OFF[seed][0][key]['task {}'.format(i + 1)][i]
            ) for i in range(1, args.tasks)]),
            np.mean([(
                OFF[seed][0][key]['task {}'.format(i + 1)][i] - EWC[seed][0][key]['task {}'.format(i + 1)][i]
            ) for i in range(1, args.tasks)]),
            np.mean([(
                OFF[seed][0][key]['task {}'.format(i + 1)][i] - OEWC[seed][0][key]['task {}'.format(i + 1)][i]
            ) for i in range(1, args.tasks)]),
            np.mean([(
                OFF[seed][0][key]['task {}'.format(i + 1)][i] - SI[seed][0][key]['task {}'.format(i + 1)][i]
            ) for i in range(1, args.tasks)]),
            np.mean([(
                OFF[seed][0][key]['task {}'.format(i + 1)][i] - NCL[seed][0][key]['task {}'.format(i + 1)][i]
            ) for i in range(1, args.tasks)]),
            np.mean([(
                OFF[seed][0][key]['task {}'.format(i + 1)][i] - KFNCL[seed][0][key]['task {}'.format(i + 1)][i]
            ) for i in range(1, args.tasks)]),
        ]


    #-------------------------------------------------------------------------------------------------#

    #--------------------#
    #----- PLOTTING -----#
    #--------------------#

    # name for plot
    plot_name = "summary-{}{}-{}".format(args.experiment, args.tasks, args.scenario)
    scheme = "{}-incremental learning".format(args.scenario)
    title = "{}  -  {}".format(args.experiment, scheme)
    
    # select names / colors / ids
    names = ["None", "Offline"]
    colors = ["grey", "black"]
    ids = [0, 1]
    names += ["EWC", "o-EWC", "SI", "NCL", "KFNCL"]
    colors += ["deepskyblue", "blue", "yellowgreen", "goldenrod", "green"]
    ids += [2,3,4,5,6]
    
    summ = {'prec': prec,
           'ave_prec': ave_prec,
           'names': names,
           'ids': ids,
           'seeds': seed_list}
    Path("./ncl_results").mkdir(parents=True, exist_ok=True)
    pickle.dump(summ, open('./ncl_results/'+plot_name+'_summary_seed_'+('_'.join([str(s) for s in seed_list]))+'.p', 'wb'))

    # open pdf
    pp = visual_plt.open_pdf("{}/{}.pdf".format(args.p_dir, plot_name))
    figure_list = []


    ########### AVERAGE ACCURACY ###########

    # bar-plot
    means = [np.mean([ave_prec[seed][id] for seed in seed_list]) for id in ids]
    if len(seed_list)>1:
        sems = [np.sqrt(np.var([ave_prec[seed][id] for seed in seed_list])/(len(seed_list)-1)) for id in ids]
        cis = [1.96*np.sqrt(np.var([ave_prec[seed][id] for seed in seed_list])/(len(seed_list)-1)) for id in ids]
    figure = visual_plt.plot_bar(means, names=names, colors=colors, ylabel="average precision (after all tasks)",
                                 title=title, yerr=cis if len(seed_list)>1 else None, ylim=(0,1))
    figure_list.append(figure)

    # print results to screen
    print("\n\n"+"#"*60+"\nSUMMARY RESULTS: {}\n".format(title)+"-"*60)
    for i,name in enumerate(names):
        if len(seed_list) > 1:
            print("{:19s} {:.2f}  (+/- {:.2f}),  n={}".format(name, 100*means[i], 100*sems[i], len(seed_list)))
        else:
            print("{:19s} {:.2f}".format(name, 100*means[i]))
        if i==1:
            print("-"*60)
    print("#"*60)

    # line-plot
    x_axes = NONE[args.seed][0]["x_task"]
    ave_lines = []
    sem_lines = []
    for id in ids:
        new_ave_line = []
        new_sem_line = []
        for line_id in range(len(prec[args.seed][id])):
            all_entries = [prec[seed][id][line_id] for seed in seed_list]
            new_ave_line.append(np.mean(all_entries))
            if len(seed_list) > 1:
                new_sem_line.append(1.96*np.sqrt(np.var(all_entries)/(len(all_entries)-1)))
        ave_lines.append(new_ave_line)
        sem_lines.append(new_sem_line)
    figure = visual_plt.plot_lines(ave_lines, x_axes=x_axes, line_names=names, colors=colors, title=title,
                                   xlabel="# of tasks", ylabel="Average precision (on tasks seen so far)",
                                   list_with_errors=sem_lines if len(seed_list)>1 else None)
    figure_list.append(figure)


    ########### BWT & FWT ###########

    # bar-plot BWT
    means = [np.mean([ave_BWT[seed][id] for seed in seed_list]) for id in ids]
    if len(seed_list) > 1:
        sems = [np.sqrt(np.var([ave_BWT[seed][id] for seed in seed_list]) / (len(seed_list) - 1)) for id in ids]
        cis = [1.96 * np.sqrt(np.var([ave_BWT[seed][id] for seed in seed_list]) / (len(seed_list) - 1)) for id in ids]
    figure = visual_plt.plot_bar(means, names=names, colors=colors, ylabel="Average BWT",
                                 title=title, yerr=cis if len(seed_list) > 1 else None)
    figure_list.append(figure)

    # bar-plot FWT
    means_fwt = [np.mean([ave_FWT[seed][id] for seed in seed_list]) for id in ids]
    if len(seed_list) > 1:
        sems_fwt = [np.sqrt(np.var([ave_FWT[seed][id] for seed in seed_list]) / (len(seed_list) - 1)) for id in ids]
        cis_fwt = [1.96 * np.sqrt(np.var([ave_FWT[seed][id] for seed in seed_list]) / (len(seed_list)-1)) for id in ids]
    figure = visual_plt.plot_bar(means_fwt, names=names, colors=colors, ylabel="Average FWT",
                                 title=title, yerr=cis_fwt if len(seed_list) > 1 else None)
    figure_list.append(figure)

    # print results to screen
    print("\n\n"+"#"*49)
    print(" "*21+"BWT              FWT\n"+"-"*49)
    for i,name in enumerate(names):
        if len(seed_list) > 1:
            print("{:16s} {:5.2f} ({:.2f})     {:5.2f} ({:.2f})".format(
                name, means[i], sems[i], means_fwt[i], sems_fwt[i],
            ))
        else:
            print("{:16s}    {:5.2f}            {:5.2f}".format(
                name, means[i], means_fwt[i]
            ))
    print("#"*49)


    ########### Forgetting & Intransigence ###########

    # bar-plot Forgetting
    means = [np.mean([ave_F[seed][id] for seed in seed_list]) for id in ids]
    if len(seed_list) > 1:
        sems = [np.sqrt(np.var([ave_F[seed][id] for seed in seed_list]) / (len(seed_list) - 1)) for id in ids]
        cis = [1.96 * np.sqrt(np.var([ave_F[seed][id] for seed in seed_list]) / (len(seed_list) - 1)) for id in ids]
    figure = visual_plt.plot_bar(means, names=names, colors=colors, ylabel="Average Forgetting",
                                 title=title, yerr=cis if len(seed_list) > 1 else None)
    figure_list.append(figure)

    # bar-plot Intransigence
    means_I = [np.mean([ave_I[seed][id] for seed in seed_list]) for id in ids]
    if len(seed_list) > 1:
        sems_I = [np.sqrt(np.var([ave_I[seed][id] for seed in seed_list]) / (len(seed_list) - 1)) for id in ids]
        cis_I = [1.96 * np.sqrt(np.var([ave_I[seed][id] for seed in seed_list]) / (len(seed_list) - 1)) for id in ids]
    figure = visual_plt.plot_bar(means_I, names=names, colors=colors, ylabel="Average Intransigence",
                                 title=title, yerr=cis_I if len(seed_list) > 1 else None)
    figure_list.append(figure)

    # print results to screen
    print("\n\n"+"#"*49)
    print(" "*17+" Forgetting      Intransigence\n"+"-"*49)
    for i,name in enumerate(names):
        if len(seed_list) > 1:
            print("{:16s} {:5.2f} ({:.2f})     {:5.2f} ({:.2f})".format(
                name, means[i], sems[i], means_I[i], sems_I[i],
            ))
        else:
            print("{:16s}    {:5.2f}            {:5.2f}".format(
                name, means[i], means_I[i]
            ))
    print("#"*49)


    # add all figures to pdf
    for figure in figure_list:
        pp.savefig(figure)

    # close the pdf
    pp.close()

    # Print name of generated plot on screen
    print("\nGenerated plot: {}/{}.pdf\n".format(args.p_dir, plot_name))