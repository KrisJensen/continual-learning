
def set_default_values(args, also_hyper_params=True):
    # -set default-values for certain arguments based on chosen scenario & experiment
    args.tasks= (5 if args.experiment=='splitMNIST' else 10) if args.tasks is None else args.tasks
    args.iters = (2000 if args.experiment=='splitMNIST' else 5000) if args.iters is None else args.iters
    args.lr = (0.001 if args.experiment=='splitMNIST' else 0.0001) if args.lr is None else args.lr
    args.fc_units = (400 if args.experiment=='splitMNIST' else 1000) if args.fc_units is None else args.fc_units
    if also_hyper_params:
        
        args.data_size = 12000 if args.data_size is None else args.data_size
        args.alpha = 1e-10 if args.alpha is None else args.alpha
        args.momentum = 0.9 if args.momentum is None else args.momentum
        args.kfncl_lr = 5e-2
        
        if args.scenario=='task':
            args.gating_prop = (
                0.95 if args.experiment == 'splitMNIST' else 0.55
            ) if args.gating_prop is None else args.gating_prop
            args.si_c = (50. if args.experiment == 'splitMNIST' else 5.) if args.si_c is None else args.si_c
            args.ewc_lambda = (
                10000000. if args.experiment == 'splitMNIST' else 500.
            ) if args.ewc_lambda is None else args.ewc_lambda
            if hasattr(args, 'o_lambda'):
                args.o_lambda = (
                    100000000. if args.experiment == 'splitMNIST' else 500.
                ) if args.o_lambda is None else args.o_lambda
            args.gamma = (0.8 if args.experiment == 'splitMNIST' else 0.8) if args.gamma is None else args.gamma
            
            if hasattr(args, 'kfac_lambda'):
                args.kfac_lambda = (1e7 if args.experiment == 'splitMNIST' else 1e0) if args.kfac_lambda is None else args.kfac_lambda
                
            if hasattr(args, 'owm_alpha'):
                args.owm_alpha = (1e-2 if args.experiment == 'splitMNIST' else 1e-2) if args.owm_alpha is None else args.owm_alpha
            
        elif args.scenario=='domain':
            args.si_c = (500. if args.experiment == 'splitMNIST' else 5.) if args.si_c is None else args.si_c
            args.ewc_lambda = (
                1000000. if args.experiment == 'splitMNIST' else 500.
            ) if args.ewc_lambda is None else args.ewc_lambda
            if hasattr(args, 'o_lambda'):
                args.o_lambda = (
                    100000000. if args.experiment == 'splitMNIST' else 1000.
                ) if args.o_lambda is None else args.o_lambda
            args.gamma = (0.7 if args.experiment == 'splitMNIST' else 0.9) if args.gamma is None else args.gamma
            
            if hasattr(args, 'kfac_lambda'):
                args.kfac_lambda = (1e6 if args.experiment == 'splitMNIST' else 1e0) if args.kfac_lambda is None else args.kfac_lambda
                
            if hasattr(args, 'owm_alpha'):
                args.owm_alpha = (1e-1 if args.experiment == 'splitMNIST' else 1e-2) if args.owm_alpha is None else args.owm_alpha
            
        elif args.scenario=='class':
            args.si_c = (0.5 if args.experiment == 'splitMNIST' else 0.1) if args.si_c is None else args.si_c
            args.ewc_lambda = (
                100000000. if args.experiment == 'splitMNIST' else 1.
            ) if args.ewc_lambda is None else args.ewc_lambda
            if hasattr(args, 'o_lambda'):
                args.o_lambda = (
                    1000000000. if args.experiment == 'splitMNIST' else 5.
                ) if args.o_lambda is None else args.o_lambda
            args.gamma = (0.8 if args.experiment == 'splitMNIST' else 1.) if args.gamma is None else args.gamma
            
            if hasattr(args, 'kfac_lambda'):
                args.kfac_lambda = (1e5 if args.experiment == 'splitMNIST' else 1e0) if args.kfac_lambda is None else args.kfac_lambda
                
            if hasattr(args, 'owm_alpha'):
                args.owm_alpha = (1e0 if args.experiment == 'splitMNIST' else 1e-2) if args.owm_alpha is None else args.owm_alpha
            
    return args
