import abc
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Categorical
from linear_nets import MLP, fc_layer
import utils


class ContinualLearner(nn.Module, metaclass=abc.ABCMeta):
    '''Abstract module to add continual learning capabilities to a classifier.

    Adds methods for "context-dependent gating" (XdG), "elastic weight consolidation" (EWC) and
    "synaptic intelligence" (SI) to its subclasses.'''
    def __init__(self):
        super().__init__()

        # XdG:
        self.mask_dict = None  # -> <dict> with task-specific masks for each hidden fully-connected layer
        self.excit_buffer_list = [
        ]  # -> <list> with excit-buffers for all hidden fully-connected layers

        # -SI:
        self.si_c = 0  #-> hyperparam: how strong to weigh SI-loss ("regularisation strength")
        self.epsilon = 0.1  #-> dampening parameter: bounds 'omega' when squared parameter-change goes to 0

        # -EWC:
        self.ewc_lambda = 0  #-> hyperparam: how strong to weigh EWC-loss ("regularisation strength")
        self.gamma = 1.  #-> hyperparam (online EWC): decay-term for old tasks' contribution to quadratic term
        self.online = True  #-> "online" (=single quadratic term) or "offline" (=quadratic term per task) EWC
        self.fisher_n = None  #-> sample size for estimating FI-matrix (if "None", full pass over dataset)
        self.emp_FI = False  #-> if True, use provided labels to calculate FI ("empirical FI"); else predicted labels
        self.EWC_task_count = 0  #-> keeps track of number of quadratic loss terms (for "offline EWC")

    def _device(self):
        return next(self.parameters()).device

    def _is_on_cuda(self):
        return next(self.parameters()).is_cuda

    @abc.abstractmethod
    def forward(self, x):
        pass

    #----------------- XdG-specifc functions -----------------#

    def apply_XdGmask(self, task):
        '''Apply task-specific mask, by setting activity of pre-selected subset of nodes to zero.

        [task]   <int>, starting from 1'''

        assert self.mask_dict is not None
        torchType = next(self.parameters()).detach()

        # Loop over all buffers for which a task-specific mask has been specified
        for i, excit_buffer in enumerate(self.excit_buffer_list):
            gating_mask = np.repeat(1., len(excit_buffer))
            gating_mask[self.mask_dict[task]
                        [i]] = 0.  # -> find task-specifc mask
            excit_buffer.set_(torchType.new(gating_mask))  # -> apply this mask

    def reset_XdGmask(self):
        '''Remove task-specific mask, by setting all "excit-buffers" to 1.'''
        torchType = next(self.parameters()).detach()
        for excit_buffer in self.excit_buffer_list:
            gating_mask = np.repeat(
                1., len(excit_buffer
                        ))  # -> define "unit mask" (i.e., no masking at all)
            excit_buffer.set_(
                torchType.new(gating_mask))  # -> apply this unit mask

    #----------------- EWC-specifc functions -----------------#

    def initialize_fisher(self):
        # initialize fisher matrix with the prior precision (c.f. NCL)
        #print('initializing fisher', self.EWC_task_count)
        assert self.online
        for n, p in self.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')
                # -take initial parameters as zero for regularization purposes
                self.register_buffer(
                    '{}_EWC_prev_task'.format(n),
                    p.detach().clone()*0)
                # -precision (approximated by diagonal Fisher Information matrix)
                self.register_buffer(
                    '{}_EWC_estimated_fisher'.format(n),
                    torch.ones(p.shape) / self.data_size)
                #print('{}_EWC_estimated_fisher'.format(n))

    def initialize_kfac_fisher(self):
        if not self.online:
            raise NotImplemented
        if not hasattr(self, 'fcE'):
            raise NotImplemented
        if not isinstance(self.fcE, MLP):
            raise NotImplemented
        if not hasattr(self, 'classifier'):
            raise NotImplemented
        if not isinstance(self.classifier, fc_layer):
            raise NotImplemented

        fcE = self.fcE
        assert fcE.kfac
        classifier = self.classifier

        def initialize_for_fcLayer(layer):
            if not isinstance(layer, fc_layer):
                raise NotImplemented
            linear = layer.linear
            g_dim, a_dim = linear.weight.shape
            abar_dim = a_dim + 1 if linear.bias is not None else a_dim
            A = torch.eye(abar_dim) / np.sqrt(self.data_size)
            G = torch.eye(g_dim) / np.sqrt(self.data_size)
            if linear.bias is None:
                bias = None
            else:
                bias = linear.bias.data
            return {'A': A, 'G': G, 'weight': linear.weight.data*0, 'bias': bias*0}

        def initialize():
            est_fisher_info = {}
            for i in range(1, fcE.layers + 1):
                label = f"fcLayer{i}"
                layer = getattr(fcE, label)
                est_fisher_info[label] = initialize_for_fcLayer(layer)
            est_fisher_info['classifier'] = initialize_for_fcLayer(classifier)
            return est_fisher_info

        self.KFAC_FISHER_INFO = initialize()

    def estimate_fisher(self, dataset, allowed_classes=None, collate_fn=None):
        '''After completing training on a task, estimate diagonal of Fisher Information matrix.

        [dataset]:          <DataSet> to be used to estimate FI-matrix
        [allowed_classes]:  <list> with class-indeces of 'allowed' or 'active' classes'''

        # Prepare <dict> to store estimated Fisher Information matrix
        est_fisher_info = {}
        for n, p in self.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')
                est_fisher_info[n] = p.detach().clone().zero_()

        # Set model to evaluation mode
        mode = self.training
        self.eval()

        # Create data-loader to give batches of size 1
        data_loader = utils.get_data_loader(dataset,
                                            batch_size=1,
                                            cuda=self._is_on_cuda(),
                                            collate_fn=collate_fn)

        # Estimate the FI-matrix for [self.fisher_n] batches of size 1
        for index, (x, y) in enumerate(data_loader):
            # break from for-loop if max number of samples has been reached
            if self.fisher_n is not None:
                if index >= self.fisher_n:
                    break
            # run forward pass of model
            x = x.to(self._device())
            output = self(x) if allowed_classes is None else self(
                x)[:, allowed_classes]
            if self.emp_FI:
                # -use provided label to calculate loglikelihood --> "empirical Fisher":
                label = torch.LongTensor([y]) if type(y) == int else y
                if allowed_classes is not None:
                    label = [
                        int(np.where(i == allowed_classes)[0][0])
                        for i in label.numpy()
                    ]
                    label = torch.LongTensor(label)
                label = label.to(self._device())
            else:
                # -use predicted label to calculate loglikelihood:
                #label = output.argmax(1)
                # TODO: needs fixing
                dist = Categorical(logits=F.log_softmax(output, dim=1))
                label = dist.sample().detach()  # do not differentiate through

            # calculate negative log-likelihood
            negloglikelihood = F.nll_loss(F.log_softmax(output, dim=1), label)

            # Calculate gradient of negative loglikelihood
            self.zero_grad()
            negloglikelihood.backward()

            # Square gradients and keep running sum
            for n, p in self.named_parameters():
                if p.requires_grad:
                    n = n.replace('.', '__')
                    if p.grad is not None:
                        est_fisher_info[n] += p.grad.detach()**2

        # Normalize by sample size used for estimation
        est_fisher_info = {n: p / index for n, p in est_fisher_info.items()}

        # Store new values in the network
        for n, p in self.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')
                # -mode (=MAP parameter estimate)
                self.register_buffer(
                    '{}_EWC_prev_task{}'.format(
                        n, "" if self.online else self.EWC_task_count + 1),
                    p.detach().clone())
                # -precision (approximated by diagonal Fisher Information matrix)
                if self.online and (self.EWC_task_count == 1
                                    or self.ncl):  #start from prior in NCL
                    existing_values = getattr(
                        self, '{}_EWC_estimated_fisher'.format(n))
                    est_fisher_info[n] += self.gamma * existing_values
                self.register_buffer(
                    '{}_EWC_estimated_fisher{}'.format(
                        n, "" if self.online else self.EWC_task_count + 1),
                    est_fisher_info[n])

        # If "offline EWC", increase task-count (for "online EWC", set it to 1 to indicate EWC-loss can be calculated)
        self.EWC_task_count = 1 if self.online else self.EWC_task_count + 1

        # Set model back to its initial mode
        self.train(mode=mode)

    def ewc_loss(self):
        '''Calculate EWC-loss.'''
        if self.EWC_task_count > 0 and not self.kfncl:
            losses = []
            # If "offline EWC", loop over all previous tasks (if "online EWC", [EWC_task_count]=1 so only 1 iteration)
            for task in range(1, self.EWC_task_count + 1):
                for n, p in self.named_parameters():
                    if p.requires_grad:
                        # Retrieve stored mode (MAP estimate) and precision (Fisher Information matrix)
                        n = n.replace('.', '__')
                        mean = getattr(
                            self, '{}_EWC_prev_task{}'.format(
                                n, "" if self.online else task))
                        fisher = getattr(
                            self, '{}_EWC_estimated_fisher{}'.format(
                                n, "" if self.online else task))
                        # If "online EWC", apply decay-term to the running sum of the Fisher Information matrices
                        fisher = self.gamma * fisher if self.online else fisher
                        # Calculate EWC-loss
                        losses.append((fisher * (p - mean)**2).sum())
            # Sum EWC-loss from all parameters (and from all tasks, if "offline EWC")
            return (1. / 2) * sum(losses)
        else:
            # EWC-loss is 0 if there are no stored mode and precision yet
            return torch.tensor(0., device=self._device())

    def estimate_kfac_fisher(self,
                             dataset,
                             allowed_classes=None,
                             collate_fn=None):
        '''After completing training on a task, estimate KFAC Fisher Information matrix.

        [dataset]:          <DataSet> to be used to estimate FI-matrix
        [allowed_classes]:  <list> with class-indeces of 'allowed' or 'active' classes'''

        if not self.online:
            raise NotImplemented
        if not hasattr(self, 'fcE'):
            raise NotImplemented
        if not isinstance(self.fcE, MLP):
            raise NotImplemented
        if not hasattr(self, 'classifier'):
            raise NotImplemented
        if not isinstance(self.classifier, fc_layer):
            raise NotImplemented

        fcE = self.fcE
        assert fcE.kfac
        classifier = self.classifier

        def initialize_for_fcLayer(layer):
            if not isinstance(layer, fc_layer):
                raise NotImplemented
            linear = layer.linear
            g_dim, a_dim = linear.weight.shape
            abar_dim = a_dim + 1 if linear.bias is not None else a_dim
            A = torch.zeros(abar_dim, abar_dim)
            G = torch.zeros(g_dim, g_dim)
            if linear.bias is None:
                bias = None
            else:
                bias = linear.bias.data.clone()
            return {
                'A': A,
                'G': G,
                'weight': linear.weight.data.clone(),
                'bias': bias
            }

        def initialize():
            est_fisher_info = {}
            for i in range(1, fcE.layers + 1):
                label = f"fcLayer{i}"
                layer = getattr(fcE, label)
                est_fisher_info[label] = initialize_for_fcLayer(layer)
            est_fisher_info['classifier'] = initialize_for_fcLayer(classifier)
            return est_fisher_info

        def update_fisher_info_layer(est_fisher_info, intermediate, label,
                                     layer, n_samples):
            if not isinstance(layer, fc_layer):
                raise NotImplemented
            if layer.phantom is None:
                raise Exception(f'Layer {label} phantom is None')
            g = layer.phantom.grad.detach()
            G = g[..., None] @ g[..., None, :]
            _a = intermediate[label].detach()
            # Here we do one batch at a time (not ideal)
            assert (_a.shape[0] == 1)
            a = _a[0]

            # check that we get the right gradients this way
            #def check():
            #    _weight_grad = g.outer(a)
            #    weight_grad = layer.linear.weight.grad
            #    return torch.allclose(weight_grad, _weight_grad)

            #assert check()

            if classifier.bias is None:
                abar = a
            else:
                o = torch.ones(*a.shape[0:-1], 1).to(self._device())
                abar = torch.cat((a, o), -1)
            A = abar[..., None] @ abar[..., None, :]
            Ao = est_fisher_info[label]['A'].to(self._device())
            Go = est_fisher_info[label]['G'].to(self._device())
            est_fisher_info[label]['A'] = Ao + A / n_samples
            est_fisher_info[label]['G'] = Go + G / n_samples

        def update_fisher_info(est_fisher_info, intermediate, n_samples):
            for i in range(1, fcE.layers + 1):
                label = f"fcLayer{i}"
                layer = getattr(fcE, label)
                update_fisher_info_layer(est_fisher_info, intermediate, label,
                                         layer, n_samples)
            update_fisher_info_layer(est_fisher_info, intermediate,
                                     'classifier', self.classifier, n_samples)

        # initialize estimated fisher info
        est_fisher_info = initialize()
        # Set model to evaluation mode
        mode = self.training
        self.eval()

        # Create data-loader to give batches of size 1
        data_loader = utils.get_data_loader(dataset,
                                            batch_size=1,
                                            cuda=self._is_on_cuda(),
                                            collate_fn=collate_fn)

        if self.fisher_n is None:
            n_samples = len(data_loader)
        else:
            n_samples = self.fisher_n

        # Estimate the FI-matrix for [self.fisher_n] batches of size 1
        for i, (x, _) in enumerate(data_loader):
            if i > n_samples:
                break
            # run forward pass of model
            x = x.to(self._device())
            if allowed_classes is None:
                output, intermediate = self(x, return_intermediate=True)
            else:
                _output, intermediate = self(x, return_intermediate=True)
                output = _output[:, allowed_classes]
            if self.emp_FI:
                raise NotImplemented
            else:
                # -use predicted label to calculate loglikelihood:
#                 label = output.argmax(1)
                dist = Categorical(logits=F.log_softmax(output, dim=1))
                label = dist.sample().detach()  # do not differentiate through

            # calculate negative log-likelihood
            negloglikelihood = F.nll_loss(F.log_softmax(output, dim=1), label)

            # Calculate gradient of negative loglikelihood
            self.zero_grad()
            negloglikelihood.backward()
            update_fisher_info(est_fisher_info, intermediate, n_samples)

        for label in est_fisher_info:
            An = est_fisher_info[label]['A'].to(self._device()) #new kronecker factor
            Gn = est_fisher_info[label]['G'].to(self._device())
            Ao = self.gamma * self.KFAC_FISHER_INFO[label]['A'].to(self._device()) #old kronecker factor
            Go = self.KFAC_FISHER_INFO[label]['G'].to(self._device()) #old kronecker factor

            As, Gs = utils.additive_nearest_kf({'A': Ao, 'G': Go}, {'A': An, 'G': Gn}) #sum of kronecker factors
            self.KFAC_FISHER_INFO[label]['A'] = As
            self.KFAC_FISHER_INFO[label]['G'] = Gs
            
#             # TODO: cook up crazy sum here
#             for k in ['A', 'G']:
#                 o = self.KFAC_FISHER_INFO[label][k].to(self._device())
#                 n = est_fisher_info[label][k].to(self._device())
#                 self.KFAC_FISHER_INFO[label][k] = o + self.gamma * n
            for param_name in ['weight', 'bias']:
                p = est_fisher_info[label][param_name].to(self._device())
                self.KFAC_FISHER_INFO[label][param_name] = p

        self.EWC_task_count = 1

        # Set model back to its initial mode
        self.train(mode=mode)

    def ewc_kfac_loss(self):
        if not self.online:
            raise NotImplemented
        if not hasattr(self, 'fcE'):
            raise NotImplemented
        if not isinstance(self.fcE, MLP):
            raise NotImplemented
        fcE = self.fcE
        assert fcE.kfac
        if not hasattr(self, 'classifier'):
            raise NotImplemented
        if not isinstance(self.classifier, fc_layer):
            raise NotImplemented

        def loss_for_layer(label, layer):
            if not isinstance(layer, fc_layer):
                raise NotImplemented
            info = self.KFAC_FISHER_INFO[label]
            A = info['A'].detach().to(self._device())
            G = info['G'].detach().to(self._device())
            bias0 = info['bias']
            weight0 = info['weight']
            bias = layer.linear.bias
            weight = layer.linear.weight
            if bias0 is not None and bias is not None:
                p = torch.cat([weight, bias[..., None]], -1)
                p0 = torch.cat([weight0, bias0[..., None]], -1)
            else:
                p = weight
                p0 = weight0
            assert (p.shape[-1] == A.shape[1])
            assert (p0.shape[-1] == A.shape[1])
            dp = p.to(self._device()) - p0.to(self._device())
            return torch.sum(dp * (G @ dp @ A))

        classifier = self.classifier
        if self.EWC_task_count > 0:
            l = loss_for_layer('classifier', classifier)
            for i in range(1, fcE.layers + 1):
                label = f"fcLayer{i}"
                nl = loss_for_layer(label, getattr(fcE, label))
                l += nl
            return 0.5 * l
        else:
            return torch.tensor(0., device=self._device())

    #------------- "Synaptic Intelligence Synapses"-specifc functions -------------#

    def update_omega(self, W, epsilon):
        '''After completing training on a task, update the per-parameter regularization strength.

        [W]         <dict> estimated parameter-specific contribution to changes in total loss of completed task
        [epsilon]   <float> dampening parameter (to bound [omega] when [p_change] goes to 0)'''

        # Loop over all parameters
        for n, p in self.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')

                # Find/calculate new values for quadratic penalty on parameters
                p_prev = getattr(self, '{}_SI_prev_task'.format(n))
                p_current = p.detach().clone()
                p_change = p_current - p_prev
                omega_add = W[n] / (p_change**2 + epsilon)
                try:
                    omega = getattr(self, '{}_SI_omega'.format(n))
                except AttributeError:
                    omega = p.detach().clone().zero_()
                omega_new = omega + omega_add

                # Store these new values in the model
                self.register_buffer('{}_SI_prev_task'.format(n), p_current)
                self.register_buffer('{}_SI_omega'.format(n), omega_new)

    def surrogate_loss(self):
        '''Calculate SI's surrogate loss.'''
        try:
            losses = []
            for n, p in self.named_parameters():
                if p.requires_grad:
                    # Retrieve previous parameter values and their normalized path integral (i.e., omega)
                    n = n.replace('.', '__')
                    prev_values = getattr(self, '{}_SI_prev_task'.format(n))
                    omega = getattr(self, '{}_SI_omega'.format(n))
                    # Calculate SI's surrogate loss, sum over all parameters
                    losses.append((omega * (p - prev_values)**2).sum())
            return sum(losses)
        except AttributeError:
            # SI-loss is 0 if there is no stored omega yet
            return torch.tensor(0., device=self._device())
