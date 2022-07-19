import collections
import itertools
from pickle import NONE
import random
import contextlib
import os
from abc import ABCMeta, abstractmethod, abstractproperty
from sys import float_repr_style
from typing import IO, Any, List, Optional, Sequence, Tuple
import ipdb
from numpy.core.fromnumeric import squeeze
from rnd_utils import RunningMeanStd
from memories import LocalMemory
import torch
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn 
from torch.nn.utils.rnn import (
    pad_packed_sequence,
    pad_sequence, 
    pack_sequence,
    pack_padded_sequence
    
    )
import pfrl
from pfrl import agent
from pfrl.utils.batch_states import batch_states
from pfrl.utils.mode_of_distribution import mode_of_distribution
from pfrl.utils.recurrent import (
    concatenate_recurrent_states,
    flatten_sequences_time_first,
    get_recurrent_state_at,
    mask_recurrent_state_at,
    one_step_forward,
    pack_and_forward,
    unwrap_packed_sequences_recursive
)
from explorer import RNDModel
import copy
import gc
from torch_win_rate import LSTM_Predictor
from rnd_utils import RewardForwardFilter
import joblib



def _mean_or_nan(xs):
    """Return its mean a non-empty sequence, numpy.nan for a empty one."""
    return np.mean(xs) if xs else np.nan


def _elementwise_clip(x, x_min, x_max):
    """Elementwise clipping

    Note: torch.clamp supports clipping to constant intervals
    """
    return torch.min(torch.max(x, x_min), x_max)


def _add_advantage_and_value_target_to_episode(
    episode, gamma, lambd, 
    rnd_normalizer: RunningMeanStd,
    reward_normalizer: RunningMeanStd    
    ):
    """Add advantage and value target values to an episode."""
    
    adv = 0.0
    int_adv = 0.0
    int_lambd = 0.95
    int_gamma = 0.99
    int_noneterminals = 1
    #print(np.sqrt(rnd_normalizer.var))
    for transition in reversed(episode):
        # external reward
        
        if reward_normalizer is not None:
            transition["reward"] = (transition["reward"]-reward_normalizer.mean)/np.sqrt(reward_normalizer.var)
        td_err = (
            transition["reward"]
            + (gamma * transition["nonterminal"] * transition["next_v_pred"])
            - transition["v_pred"]
        )

        adv = td_err + gamma * lambd * adv
        transition["adv"] = adv
        transition["v_teacher"] = adv + transition["v_pred"]

        if "r_i" in transition.keys():
            # normalizing internal reward
            if rnd_normalizer is not None:
                transition["r_i"] = transition["r_i"]/np.sqrt(rnd_normalizer.var)
            

            int_td_err = (
                transition["r_i"]
                + (int_gamma * int_noneterminals * transition["next_int_v_pred"])
                - transition["int_v_pred"]
            )
            int_adv = int_td_err + int_gamma * int_lambd * int_adv
            transition["adv_i"] = int_adv
            transition["int_v_teacher"] = adv + transition["int_v_pred"]




def _add_advantage_and_value_target_to_episodes(
    episodes, gamma, lambd, rnd_normalizer: RunningMeanStd, reward_normalizer: RunningMeanStd):
    """Add advantage and value target values to a list of episodes."""
    
    for episode in episodes:
        
        _add_advantage_and_value_target_to_episode(
            episode, gamma=gamma, lambd=lambd,
            rnd_normalizer=rnd_normalizer,
            reward_normalizer=reward_normalizer
            )


def _add_log_prob_and_value_to_episodes_recurrent(
    episodes,
    model,
    phi,
    batch_states,
    obs_normalizer,
    device,
):
    # Sort desc by lengths so that pack_sequence does not change the order
    episodes = sorted(episodes, key=len, reverse=True)

    # Prepare data for a recurrent model
    seqs_states = []
    seqs_next_states = []
    seqs_lens = []
    seqs_prev_actions = []
    seqs_prev_actions_of_next_states = []
    
    for ep in episodes:
        if "battle_result" in ep[0].keys():
            win_label =[
                int(transition["battle_result"][0]) for transition in ep 
                if transition["battle_result"][0] is not None
                ]
            for transition in ep:
                transition["battle_result"] = win_label
            
        states = batch_states(
            [transition["state"] for transition in ep],
            device, 
            phi
        )
        next_states = batch_states(
            [transition["next_state"] for transition in ep], 
            device, 
            phi
        )
        
        # prev act
        prev_actions = torch.unsqueeze(
            batch_states(
                [
                    transition["prev_action"] if transition["prev_action"] is not None else 0 for transition in ep ],
                device,
                phi,
            ),
            dim = -1
        )
        
        onehot_prev_actions = torch.zeros(
            (len(prev_actions), 14)).to(device).scatter_(1, prev_actions.to(device), 1.0)
                
        onehot_prev_actions[0][0] = 0
        # prev action of next state
        prev_actions_of_next_states = torch.unsqueeze(
            batch_states(
                [transition["action"] for transition in ep],
                device,
                phi,
            ),
            dim = -1
        ).to(device)
        onehot_prev_actions_of_next_states = torch.zeros(
            (
                len(prev_actions_of_next_states), 14)
                ).to(device).scatter_(1, prev_actions_of_next_states.to(device), 1.0
            )
        if obs_normalizer:
            states = obs_normalizer(states.cpu()).to(device)
            next_states = obs_normalizer(next_states.cpu()).to(device)
        
        
        seqs_states.append(states)
        seqs_next_states.append(next_states)
        seqs_prev_actions.append(onehot_prev_actions)
        seqs_prev_actions_of_next_states.append(onehot_prev_actions_of_next_states)
        seqs_lens.append(len(states))
        
    

    flat_transitions = flatten_sequences_time_first(episodes)

    # Predict values using a recurrent model
    with torch.no_grad(), pfrl.utils.evaluating(model):
        rs = concatenate_recurrent_states([ep[0]["recurrent_state"] for ep in episodes])
        next_rs = concatenate_recurrent_states(
            [ep[0]["next_recurrent_state"] for ep in episodes]
        )
        assert (rs is None) or (next_rs is None) or (len(rs) == len(next_rs))
        
        '''
        (flat_distribs, flat_vs), _ = pack_and_forward(model, seqs_states, rs)
        (_, flat_next_vs), _ = pack_and_forward(model, seqs_next_states, next_rs)
        '''
        if rs is not None:
            new_rs = []
            new_rs.append(rs[0].to(device) )
            new_rs.append(rs[1].to(device) )
            rs = tuple(new_rs)
            
        if next_rs is not None:
            new_next_rs = []
            new_next_rs.append(next_rs[0].to(device) )
            new_next_rs.append(next_rs[1].to(device) )
            next_rs = tuple(new_next_rs)
            
        #import ipdb; ipdb.set_trace()
        (probs, vs, int_vs), _ = pack_and_forward(model, seqs_states, seqs_prev_actions, rs)
        (_, next_vs, next_int_vs), _ = pack_and_forward(model, seqs_next_states, seqs_prev_actions_of_next_states,next_rs)
        
        padded_logits = pack_padded_sequence(probs, seqs_lens).data
        padded_vs = pack_padded_sequence(vs, seqs_lens).data
        padded_next_vs = pack_padded_sequence(next_vs, seqs_lens).data


        flat_distribs = pfrl.policies.SoftmaxCategoricalHead(
        )(torch.reshape(padded_logits, (-1, 14)))
        
        flat_vs = torch.reshape(padded_vs, (-1, 1))
        flat_next_vs = torch.reshape(padded_next_vs, (-1, 1))
      
        
        flat_actions = torch.tensor(
            [b["action"] for b in flat_transitions], device=device
        )
        
        
        flat_log_probs = torch.unsqueeze(
            flat_distribs.log_prob(torch.squeeze(flat_actions)), -1).cpu().numpy()
        flat_vs = flat_vs.cpu().numpy()
        flat_next_vs = flat_next_vs.cpu().numpy()


        # Intrinsic reward
        if int_vs is not None:
            padded_int_vs = pack_padded_sequence(int_vs, seqs_lens).data
            padded_next_int_vs = pack_padded_sequence(next_int_vs, seqs_lens).data
            flat_int_vs = torch.reshape(padded_int_vs, (-1, 1))
            flat_next_int_vs = torch.reshape(padded_next_int_vs, (-1, 1))
            flat_int_vs = flat_int_vs.cpu().numpy()
            flat_next_int_vs = flat_next_int_vs.cpu().numpy()

    # Add predicted values to transitions
    if int_vs is not None:
        for transition, log_prob, v, next_v, int_v, next_int_v in zip(
            flat_transitions, 
            flat_log_probs, 
            flat_vs, 
            flat_next_vs,
            flat_int_vs,
            flat_next_int_vs
        ):  
            
            transition["log_prob"] = float(log_prob)
            transition["v_pred"] = float(v)
            transition["next_v_pred"] = float(next_v)
            transition["int_v_pred"] = float(int_v)
            transition["next_int_v_pred"] = float(next_int_v)


    else:
        for transition, log_prob, v, next_v in zip(
        flat_transitions, 
        flat_log_probs, 
        flat_vs, 
        flat_next_vs
        ):  
            
            transition["log_prob"] = float(log_prob)
            transition["v_pred"] = float(v)
            transition["next_v_pred"] = float(next_v)

    






def _add_log_prob_and_value_to_episodes(
    episodes,
    model,
    phi,
    batch_states,
    obs_normalizer,
    device,
):

    dataset = list(itertools.chain.from_iterable(episodes))

    # Compute v_pred and next_v_pred
    states = batch_states([b["state"] for b in dataset], device, phi)
    next_states = batch_states([b["next_state"] for b in dataset], device, phi)

    if obs_normalizer:
        states = obs_normalizer(states)
        next_states = obs_normalizer(next_states)

    with torch.no_grad(), pfrl.utils.evaluating(model):
        distribs, vs_pred = model(states)
        _, next_vs_pred = model(next_states)

        actions = torch.tensor([b["action"] for b in dataset], device=device)
        log_probs = distribs.log_prob(actions).cpu().numpy()
        vs_pred = vs_pred.cpu().numpy().ravel()
        next_vs_pred = next_vs_pred.cpu().numpy().ravel()

    for transition, log_prob, v_pred, next_v_pred in zip(
        dataset, log_probs, vs_pred, next_vs_pred
    ):
        transition["log_prob"] = log_prob
        transition["v_pred"] = v_pred
        transition["next_v_pred"] = next_v_pred


def _limit_sequence_length(sequences, max_len):
    assert max_len > 0
    new_sequences = []
    for sequence in sequences:
        while len(sequence) > max_len:
            new_sequences.append(sequence[:max_len])
            sequence = sequence[max_len:]
        assert 0 < len(sequence) <= max_len
        new_sequences.append(sequence)
    return new_sequences


def _yield_subset_of_sequences_with_fixed_number_of_items(sequences, n_items):
    assert n_items > 0
    stack = list(reversed(sequences))
    while stack:
        subset = []
        count = 0
        while count < n_items and stack:
            sequence = stack.pop()
            subset.append(sequence)
            count += len(sequence)
        if count > n_items:
            # Split last sequence
            sequence_to_split = subset[-1]
            n_exceeds = count - n_items
            assert n_exceeds > 0
            subset[-1] = sequence_to_split[:-n_exceeds]
            stack.append(sequence_to_split[-n_exceeds:])
        if sum(len(seq) for seq in subset) == n_items:
            yield subset
        else:
            # This ends the while loop.
            assert len(stack) == 0

def _yield_subset_of_sequences_with_fixed_number_of_items_with_limit(
    sequences, n_items, time_recorder, sample_limit=4
    ):
    assert n_items > 0
    
    stack = list(reversed(sequences))
    while stack:
        subset = []
        count = 0
        while count < n_items and stack:
            i, sequence = stack.pop()
            if time_recorder[i] < sample_limit:
                subset.append(sequence)
                time_recorder[i] += 1
                count += len(sequence)
        if len(stack) != 0:
            yield subset
        else:
            # This ends the while loop.
            assert len(stack) == 0

def _compute_explained_variance(transitions):
    """Compute 1 - Var[return - v]/Var[return].

    This function computes the fraction of variance that value predictions can
    explain about returns.
    """
    t = np.array([tr["v_teacher"] for tr in transitions])
    y = np.array([tr["v_pred"] for tr in transitions])
    vart = np.var(t)

    int_t = np.array([tr["int_v_teacher"] for tr in transitions])
    int_y = np.array([tr["int_v_pred"] for tr in transitions])
    int_vart = np.var(int_t)

    if vart == 0:
        return np.nan, np.nan
    else:
        return float(1 - np.var(t - y) / vart), float(1 - np.var(int_t - int_y) / int_vart)


def _make_dataset_recurrent(
    episodes,
    model,
    phi,
    batch_states,
    obs_normalizer,
    gamma,
    lambd,
    max_recurrent_sequence_len,
    rnd_normalizer: RunningMeanStd,
    reward_normalizer,
    device,
):
    """Make a list of sequences with necessary information."""

    _add_log_prob_and_value_to_episodes_recurrent(
        episodes=episodes,
        model=model,
        phi=phi,
        batch_states=batch_states,
        obs_normalizer = obs_normalizer,
        device=device,
    )
    
    
    _add_advantage_and_value_target_to_episodes(
        episodes, gamma=gamma, lambd=lambd, 
        rnd_normalizer = rnd_normalizer,
        reward_normalizer = reward_normalizer
        )

    

    if max_recurrent_sequence_len is not None:
        dataset = _limit_sequence_length(episodes, max_recurrent_sequence_len)
    else:
        dataset = list(episodes)

    return dataset


def _make_dataset(
    episodes, model, phi, batch_states, obs_normalizer, gamma, lambd, device
):
    """Make a list of transitions with necessary information."""

    _add_log_prob_and_value_to_episodes(
        episodes=episodes,
        model=model,
        phi=phi,
        batch_states=batch_states,
        obs_normalizer=obs_normalizer,
        device=device,
    )

    _add_advantage_and_value_target_to_episodes(episodes, gamma=gamma, lambd=lambd)

    return list(itertools.chain.from_iterable(episodes))


def _yield_minibatches(dataset, minibatch_size, num_epochs):
    assert dataset
    buf = []
    n = 0
    while n < len(dataset) * num_epochs:
        while len(buf) < minibatch_size:
            buf = random.sample(dataset, k=len(dataset)) + buf
        assert len(buf) >= minibatch_size
        yield buf[-minibatch_size:]
        n += minibatch_size
        buf = buf[:-minibatch_size]

from torch.optim import Optimizer
class AttributeSavingMixin(object):
    """Mixin that provides save and load functionalities."""

    @abstractproperty
    def saved_attributes(self) -> Tuple[str, ...]:
        """Specify attribute names to save or load as a tuple of str."""
        pass

    def save(self, dirname: str) -> None:
        """Save internal states."""
        self.__save(dirname, [])

    def __save(self, dirname: str, ancestors: List[Any]):
        os.makedirs(dirname, exist_ok=True)
        ancestors.append(self)
        for attr in self.saved_attributes:
            assert hasattr(self, attr)
            attr_value = getattr(self, attr)
            if attr_value is None:
                continue
            if isinstance(attr_value, AttributeSavingMixin):
                assert not any(
                    attr_value is ancestor for ancestor in ancestors
                ), "Avoid an infinite loop"
                attr_value.__save(os.path.join(dirname, attr), ancestors)
            else:
                if isinstance(
                    attr_value,
                    (torch.nn.parallel.DistributedDataParallel, torch.nn.DataParallel),
                ):
                    attr_value = attr_value.module
                
                if isinstance(attr_value, Optimizer):
                    torch.save(
                        attr_value.state_dict(), os.path.join(dirname, "{}.pt".format(attr))
                    )
                else:
                
                    torch.save(
                    attr_value.state_dict(), os.path.join(dirname, "{}.pt".format(attr))
                )
       
        ancestors.pop()
        
        

    def load(self, dirname: str) -> None:
        """Load internal states."""
        self.__load(dirname, [])

    def __load(self, dirname: str, ancestors: List[Any]) -> None:
        map_location = torch.device("cpu")
        # if not torch.cuda.is_available() else None
        ancestors.append(self)
        for attr in self.saved_attributes:
            assert hasattr(self, attr)
            attr_value = getattr(self, attr)
            if attr_value is None:
                continue
            if isinstance(attr_value, AttributeSavingMixin):
                assert not any(
                    attr_value is ancestor for ancestor in ancestors
                ), "Avoid an infinite loop"
                attr_value.load(os.path.join(dirname, attr))
            else:
                if isinstance(
                    attr_value,
                    (torch.nn.parallel.DistributedDataParallel, torch.nn.DataParallel),
                ):
                    attr_value = attr_value.module
                attr_value.load_state_dict(
                    torch.load(
                        os.path.join(dirname, "{}.pt".format(attr)), map_location
                    )
                )
        ancestors.pop()
        
    
class BatchAgent(agent.Agent, metaclass=ABCMeta):
    """Abstract agent class that can interact with a batch of envs."""

    def act(self, obs: Any) -> Any:
        return self.batch_act([obs])[0]

    def observe(self, obs: Any, reward: float, done: bool, reset: bool, battle_result) -> None:
        return self.batch_observe([obs], [reward], [done], [reset], [battle_result])

    @abstractmethod
    def batch_act(self, batch_obs: Sequence[Any]) -> Sequence[Any]:
        """Select a batch of actions.

        Args:
            batch_obs (Sequence of ~object): Observations.

        Returns:
            Sequence of ~object: Actions.
        """
        raise NotImplementedError()

    @abstractmethod
    def batch_observe(
        self,
        batch_obs: Sequence[Any],
        batch_reward: Sequence[float],
        batch_done: Sequence[bool],
        batch_reset: Sequence[bool],
    ) -> None:
        """Observe a batch of action consequences.

        Args:
            batch_obs (Sequence of ~object): Observations.
            batch_reward (Sequence of float): Rewards.
            batch_done (Sequence of boolean): Boolean values where True
                indicates the current state is terminal.
            batch_reset (Sequence of boolean): Boolean values where True
                indicates the current episode will be reset, even if the
                current state is not terminal.

        Returns:
            None
        """
        raise NotImplementedError()

class PPO(AttributeSavingMixin, BatchAgent):
    """Proximal Policy Optimization

    See https://arxiv.org/abs/1707.06347

    Args:
        model (torch.nn.Module): Model to train (including recurrent models)
            state s  |->  (pi(s, _), v(s))
        optimizer (torch.optim.Optimizer): Optimizer used to train the model
        gpu (int): GPU device id if not None nor negative
        gamma (float): Discount factor [0, 1]
        lambd (float): Lambda-return factor [0, 1]
        phi (callable): Feature extractor function
        value_func_coef (float): Weight coefficient for loss of
            value function (0, inf)
        entropy_coef (float): Weight coefficient for entropy bonus [0, inf)
        update_interval (int): Model update interval in step
        minibatch_size (int): Minibatch size
        epochs (int): Training epochs in an update
        clip_eps (float): Epsilon for pessimistic clipping of likelihood ratio
            to update policy
        clip_eps_vf (float): Epsilon for pessimistic clipping of value
            to update value function. If it is ``None``, value function is not
            clipped on updates.
        standardize_advantages (bool): Use standardized advantages on updates
        recurrent (bool): If set to True, `model` is assumed to implement
            `pfrl.nn.Recurrent` and update in a recurrent
            manner.
        max_recurrent_sequence_len (int): Maximum length of consecutive
            sequences of transitions in a minibatch for updatig the model.
            This value is used only when `recurrent` is True. A smaller value
            will encourage a minibatch to contain more and shorter sequences.
        act_deterministically (bool): If set to True, choose most probable
            actions in the act method instead of sampling from distributions.
        max_grad_norm (float or None): Maximum L2 norm of the gradient used for
            gradient clipping. If set to None, the gradient is not clipped.
        value_stats_window (int): Window size used to compute statistics
            of value predictions.
        entropy_stats_window (int): Window size used to compute statistics
            of entropy of action distributions.
        value_loss_stats_window (int): Window size used to compute statistics
            of loss values regarding the value function.
        policy_loss_stats_window (int): Window size used to compute statistics
            of loss values regarding the policy.

    Statistics:
        average_value: Average of value predictions on non-terminal states.
            It's updated on (batch_)act_and_train.
        average_entropy: Average of entropy of action distributions on
            non-terminal states. It's updated on (batch_)act_and_train.
        average_value_loss: Average of losses regarding the value function.
            It's updated after the model is updated.
        average_policy_loss: Average of losses regarding the policy.
            It's updated after the model is updated.
        n_updates: Number of model updates so far.
        explained_variance: Explained variance computed from the last batch.
    """

    saved_attributes = ("model", "optimizer")

    def __init__(
        self,
        model,
        optimizer: torch.optim.Adam =None,
        obs_normalizer: RunningMeanStd=None,
        reward_normalizer: RunningMeanStd=None,
        gpu=None,
        gamma=0.99,
        lambd=0.95,
        phi=lambda x: x,
        value_func_coef=1.0,
        entropy_coef=0.01,
        update_interval=2048,
        minibatch_size=64,
        epochs=10,
        clip_eps=0.2,
        clip_eps_vf=None,
        standardize_advantages=True,
        batch_states=batch_states,
        recurrent=False,
        max_recurrent_sequence_len=None,
        act_deterministically=False,
        max_grad_norm=5,
        value_stats_window=100,
        entropy_stats_window=100,
        value_loss_stats_window=100,
        policy_loss_stats_window=100,
        rnd_model = None,
        rnd_normalizer = None,
        memory: LocalMemory = None
    ):
        self.model = model
        self.optimizer = optimizer
        self.obs_normalizer = obs_normalizer
        if gpu is not None:
            assert torch.cuda.is_available()
            self.device = torch.device("cuda:{}".format(gpu))
            self.model.to(self.device)
            
            '''
            if self.obs_normalizer is not None:
                self.obs_normalizer.to(self.device)
            else:
                self.device = torch.device("cuda:{}".format(1))
                self.model.to(self.device)
                self.model = torch.nn.DataParallel(self.model, dim=1)
                if self.obs_normalizer is not None:
                    self.obs_normalizer.to(self.device)
            '''
            
        else:
            self.device = torch.device("cpu")

        self.gamma = gamma
        self.lambd = lambd
        self.phi = phi
        self.value_func_coef = value_func_coef
        self.entropy_coef = entropy_coef
        self.update_interval = update_interval
        self.minibatch_size = minibatch_size
        self.epochs = epochs
        self.clip_eps = clip_eps
        self.clip_eps_vf = clip_eps_vf
        self.standardize_advantages = standardize_advantages
        self.batch_states = batch_states
        self.recurrent = recurrent
        self.max_recurrent_sequence_len = max_recurrent_sequence_len
        self.act_deterministically = act_deterministically
        self.max_grad_norm = max_grad_norm
        

        # Contains episodes used for next update iteration
        self.memory = memory if memory is not None else []

        # Contains transitions of the last episode not moved to self.memory yet
        self.last_episode = []
        self.last_state = None
        self.last_action = None

        # Batch versions of last_episode, last_state, and last_action
        self.batch_last_episode = None
        self.batch_last_state = None
        self.batch_last_action = None
        self.batch_last_last_action = 0

        # Recurrent states of the model
        self.train_recurrent_states = None
        self.train_prev_recurrent_states = None
        self.test_recurrent_states = None

        self.value_record = collections.deque(maxlen=value_stats_window)
        self.entropy_record = collections.deque(maxlen=entropy_stats_window)
        self.value_loss_record = collections.deque(maxlen=value_loss_stats_window)
        self.policy_loss_record = collections.deque(maxlen=policy_loss_stats_window)
        self.len_episode_record = collections.deque(maxlen=value_stats_window)
        self.prev_action = torch.zeros(1, 14)
        self.explained_variance = np.nan
        self.int_explained_variance = np.nan
        self.n_updates = 0

        self.global_step = 0
        self.step_per_episode = 0
        
        self.gpu = gpu 
        self.prev_act = None

        self.episode_end_turn = 0
        
        self.reward_normalizer = reward_normalizer

        # for intrinsic reward compute
        self.rnd_model: RNDModel = rnd_model
        if gpu is not None and self.rnd_model:
            self.rnd_model.to(self.device)
        self.update_proportion = 0.75
        self.int_coef = 1
        self.ext_coef = 2
        self.int_value_records = collections.deque(maxlen=value_stats_window)
        self.int_value_loss_record = collections.deque(maxlen=value_loss_stats_window)
        self.rnd_loss_record = collections.deque(maxlen=value_loss_stats_window)
        self.int_reward_record= collections.deque(maxlen=value_stats_window)
        self.rnd_normalizer: RunningMeanStd = rnd_normalizer
        self.rnd_mean_record = collections.deque(maxlen=value_loss_stats_window)
        self.rnd_var_record = collections.deque(maxlen=value_loss_stats_window)
        self.num_win = 0
        self.num_training = 0
        
        # update entropy coef
        self.entropy_coef_start = 0.001
        self.entropy_coef_decay_step = 5000000
        self.entropy_coef_end = self.entropy_coef_start/10
        #self.curr_lr = self.lr_start

        self.int_gamma = 0.99
        self.ext_reward_filter = RewardForwardFilter(self.gamma)
        self.int_reward_fileter = RewardForwardFilter(self.int_gamma)

        # for early stopping
        self.target_kl = None
        self.approx_kl_div_record = collections.deque(maxlen=value_loss_stats_window)
         
        # update lr
        if self.optimizer is not None:
            
            self.lr_start = 5e-5
            self.lr_decay_step = 5000000
            self.lr_end = self.lr_start/10
            self.curr_lr = self.optimizer.defaults["lr"]
    
    def update_entropy_coef(self):

        diff = self.entropy_coef_end - self.entropy_coef_start
        new_entropy_coef = self.entropy_coef_start + (
            diff * (
                self.n_updates/self.entropy_coef_decay_step
                )
            )
        
        self.entropy_coef = new_entropy_coef

    def _initialize_batch_variables(self, num_envs):
        self.batch_last_episode = [[] for _ in range(num_envs)]
        self.batch_last_state = [None] * num_envs
        self.batch_last_action = [None] * num_envs

    def _update_if_dataset_is_ready(self):
        
        do_train = False
        
        dataset_size = (
            sum(len(episode) for episode in self.memory.memory)
            + len(self.last_episode)
            + (
                0
                if self.batch_last_episode is None
                else sum(len(episode) for episode in self.batch_last_episode)
            )
        )
        
        
        if dataset_size >= self.update_interval:
            
            do_train = True
            self._flush_last_episode()
            if self.recurrent:
                
                if self.rnd_model or self.reward_normalizer:
                    flat_data = list(itertools.chain.from_iterable(self.memory.memory))
                    if self.rnd_model is not None:
                        # update normalizer
                        print(self.int_reward_fileter.rewems)
                        int_rewards = self.batch_states([self.int_reward_fileter.update(b["r_i"]) for b in flat_data], self.device, self.phi).detach().cpu().numpy()
                        mean, var, count = np.mean(int_rewards), np.std(int_rewards)**2, len(int_rewards)
                        self.rnd_normalizer.update_from_moments(mean, var, count)
                        self.rnd_mean_record.append([mean])
                        self.rnd_var_record.append([var])
                        print(self.int_reward_fileter.rewems)
                        
                        
                    if self.reward_normalizer is not None:
                        import ipdb; ipdb.set_trace()
                        rewards = self.batch_states([b["reward"] for b in flat_data], self.device, self.phi).detach().cpu().numpy()
                        mean, var, count = np.mean(rewards), np.std(rewards)**2, len(rewards)
                        self.reward_normalizer.update_from_moments(mean, var, count)
                    
                dataset = _make_dataset_recurrent(

                    episodes=copy.deepcopy(self.memory.memory),
                    model=self.model,
                    phi=self.phi,
                    batch_states=self.batch_states,
                    obs_normalizer=self.obs_normalizer,
                    gamma=self.gamma,
                    lambd=self.lambd,
                    max_recurrent_sequence_len=self.max_recurrent_sequence_len,
                    rnd_normalizer=self.rnd_normalizer,
                    reward_normalizer = self.reward_normalizer,
                    device=self.device,
                )
                self.memory.memory = dataset
                self._update_recurrent(dataset)
            else:
                dataset = _make_dataset(
                    episodes=self.memory.memory,
                    model=self.model,
                    phi=self.phi,
                    batch_states=self.batch_states,
                    obs_normalizer=self.obs_normalizer,
                    gamma=self.gamma,
                    lambd=self.lambd,
                    device=self.device,
                )
                assert len(dataset) == dataset_size
                self._update(dataset)
            
            self.explained_variance, self.int_explained_variance  = _compute_explained_variance(
                list(itertools.chain.from_iterable(self.memory.memory))
            )
            
            self.memory.reset()
            del dataset
            gc.collect()
            
            
        return do_train

    def _flush_last_episode(self):
        if self.last_episode:
            self.memory.append(self.last_episode)
            self.last_episode = []
        if self.batch_last_episode:
            for i, episode in enumerate(self.batch_last_episode):
                if episode:
                    self.memory.append(episode)
                    self.batch_last_episode[i] = []

    def _update_obs_normalizer(self, dataset):
        assert self.obs_normalizer
        states = self.batch_states([b["state"] for b in dataset], self.device, self.phi)
        self.obs_normalizer.update(states.detach().cpu().numpy())

    def _update(self, dataset):
        """Update both the policy and the value function."""

        device = self.device

        if self.obs_normalizer:
            self._update_obs_normalizer(dataset)

        assert "state" in dataset[0]
        assert "v_teacher" in dataset[0]

        if self.standardize_advantages:
            all_advs = torch.tensor([b["adv"] for b in dataset], device=device)
            std_advs, mean_advs = torch.std_mean(all_advs, unbiased=False)

        for batch in _yield_minibatches(
            dataset, minibatch_size=self.minibatch_size, num_epochs=self.epochs
        ):
            states = self.batch_states(
                [b["state"] for b in batch], self.device, self.phi
            )
            if self.obs_normalizer:
                states = self.obs_normalizer(states)
            actions = torch.tensor([b["action"] for b in batch], device=device)
            distribs, vs_pred = self.model(states)

            advs = torch.tensor(
                [b["adv"] for b in batch], dtype=torch.float32, device=device
            )
            if self.standardize_advantages:
                advs = (advs - mean_advs) / (std_advs + 1e-8)

            log_probs_old = torch.tensor(
                [b["log_prob"] for b in batch],
                dtype=torch.float,
                device=device,
            )
            vs_pred_old = torch.tensor(
                [b["v_pred"] for b in batch],
                dtype=torch.float,
                device=device,
            )
            vs_teacher = torch.tensor(
                [b["v_teacher"] for b in batch],
                dtype=torch.float,
                device=device,
            )
            # Same shape as vs_pred: (batch_size, 1)
            vs_pred_old = vs_pred_old[..., None]
            vs_teacher = vs_teacher[..., None]

            self.model.zero_grad()
            loss = self._lossfun(
                distribs.entropy(),
                vs_pred,
                distribs.log_prob(actions),
                vs_pred_old=vs_pred_old,
                log_probs_old=log_probs_old,
                advs=advs,
                vs_teacher=vs_teacher,
            )
            loss.backward()
            if self.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )
            self.optimizer.step()
            self.n_updates += 1

    def _update_once_recurrent(
        self, episodes, mean_advs, std_advs, mean_int_advs, std_int_advs):

        assert std_advs is None or std_advs > 0



        device = self.device
        training_stop = False

        # Sort desc by lengths so that pack_sequence does not change the order
        episodes = sorted(episodes, key=len, reverse=True)

        flat_transitions = flatten_sequences_time_first(episodes)

        # Prepare data for a recurrent model
        seqs_states = []
        seqs_lens = []
        seqs_prev_actions = []
        for ep in episodes:
            states = self.batch_states(
                [transition["state"] for transition in ep],
                device,
                self.phi,
            )
            # prev act
            prev_actions = torch.unsqueeze(
                batch_states(
                    [
                    transition["prev_action"] if transition["prev_action"] is not None else 0 for transition in ep ],
                    self.device,
                    self.phi,
                ),
                dim = -1
            )
            onehot_prev_actions = torch.zeros(
                (
                    len(prev_actions), 14)
                    ).to(device).scatter_(1, prev_actions, 1.0
                )
            onehot_prev_actions[0][0] = 0

            if self.obs_normalizer:
                states = self.obs_normalizer(states.cpu()).to(device)
            seqs_states.append(states)
            seqs_prev_actions.append(onehot_prev_actions)
            seqs_lens.append(len(states))

        flat_actions = torch.tensor(
            [transition["action"] for transition in flat_transitions],
            device=device,
        )
        flat_advs = torch.tensor(
            [transition["adv"] for transition in flat_transitions],
            dtype=torch.float,
            device=device,
        )
        '''
        if self.standardize_advantages:
            flat_advs = (flat_advs - mean_advs) / (std_advs + 1e-8)
        '''
        flat_log_probs_old = torch.tensor(
            [transition["log_prob"] for transition in flat_transitions],
            dtype=torch.float,
            device=device,
        )
        flat_vs_pred_old = torch.tensor(
            [[transition["v_pred"]] for transition in flat_transitions],
            dtype=torch.float,
            device=device,
        )
        flat_vs_teacher = torch.tensor(
            [[transition["v_teacher"]] for transition in flat_transitions],
            dtype=torch.float,
            device=device,
        )
        if self.rnd_model:
            # --------------------------------------------------------------------------------
            # for Curiosity-driven(Random Network Distillation)
            # computing components
            flat_int_advs = torch.tensor(
                [transition["adv_i"] for transition in flat_transitions],
                dtype=torch.float,
                device=device,
                )

            
            
            flat_advs = flat_advs*self.ext_coef + flat_int_advs*self.int_coef
            
            if self.standardize_advantages:
                flat_advs = (flat_advs-flat_advs.mean()) /(flat_advs.std() + 1e-8)

            flat_int_vs_teacher = torch.tensor(
                [[transition["int_v_teacher"]] for transition in flat_transitions],
                dtype=torch.float,
                device=device,
            )

        with torch.no_grad(), pfrl.utils.evaluating(self.model):
            rs = concatenate_recurrent_states(
                [ep[0]["recurrent_state"] for ep in episodes]
            )

            



        self.model.zero_grad()
        
        (distribs, vs_pred, int_vs_pred), _ = pack_and_forward(self.model, seqs_states, seqs_prev_actions, rs)
        
        
        
        padded_logits = pack_padded_sequence(distribs, seqs_lens).data
        padded_vs_pred = pack_padded_sequence(vs_pred, seqs_lens).data
        flat_distribs = pfrl.policies.SoftmaxCategoricalHead(
            )(padded_logits)
    
        flat_vs_pred = padded_vs_pred
        flat_log_probs = flat_distribs.log_prob(torch.squeeze(flat_actions))
        flat_entropy = flat_distribs.entropy()        

        with torch.no_grad():
            # calculate approx_kl
            log_ratio = flat_log_probs - flat_log_probs_old
            approx_kl_div = torch.mean((torch.exp(log_ratio)-1) - log_ratio).cpu().numpy()
            self.approx_kl_div_record.append(float(approx_kl_div))
            if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                print(f"earlystopping because approx_kl({float(approx_kl_div)}) > 1.5 * {self.target_kl}")
                training_stop = True

        if training_stop is False:
            loss = self._lossfun(
                entropy=flat_entropy,
                vs_pred=flat_vs_pred,
                log_probs=flat_log_probs,
                vs_pred_old=flat_vs_pred_old,
                log_probs_old=flat_log_probs_old,
                advs=flat_advs,
                vs_teacher=flat_vs_teacher,
            )
            
            

            if self.rnd_model:
                # --------------------------------------------------------------------------------
                # for Curiosity-driven(Random Network Distillation)
                
                
                
                padded_int_vs_pred = pack_padded_sequence(int_vs_pred, seqs_lens).data
                int_value_loss = F.mse_loss(padded_int_vs_pred, flat_int_vs_teacher)

                predict_next_state_feature, target_next_state_feature = self.rnd_model(
                    torch.reshape(
                        pad_sequence(seqs_states), (-1, seqs_states[0].shape[-1])
                        ),
                    torch.reshape(
                        pad_sequence(seqs_prev_actions), (-1, seqs_prev_actions[0].shape[-1])
                    )
                    )

                
                forward_loss = self.rnd_model.loss_func(predict_next_state_feature, target_next_state_feature.detach()).mean(-1)
                # Proportion of exp used for predictor update
                mask = torch.rand(len(forward_loss)).to(self.device)
                mask = (mask < self.update_proportion).type(torch.FloatTensor).to(self.device)
                forward_loss = (forward_loss * mask).sum() / torch.max(mask.sum(), torch.Tensor([1]).to(self.device))
                
                # loss 
                self.int_value_loss_record.append(float(int_value_loss))
                self.rnd_loss_record.append(float(forward_loss[0]))
                loss += forward_loss[0]
                loss += int_value_loss
                # ---------------------------------------------------------------------------------
            
            loss.backward()
            if self.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                if self.rnd_model is not None:
                    torch.nn.utils.clip_grad_norm_(self.rnd_model.parameters(), self.max_grad_norm)
                
            self.optimizer.step()
            self.n_updates += 1
            self.update_lr()
            self.update_entropy_coef()

        return seqs_states, seqs_prev_actions, training_stop

    def _update_recurrent(self, dataset):
        """Update both the policy and the value function."""

        device = self.device

        flat_dataset = list(itertools.chain.from_iterable(dataset))
        
        if self.obs_normalizer:
            self._update_obs_normalizer(flat_dataset)
        


        assert "state" in flat_dataset[0]
        assert "v_teacher" in flat_dataset[0]

        
        
        mean_advs = None
        std_advs = None
        std_int_advs = None
        mean_int_advs = None
        '''
        if self.standardize_advantages:
        all_advs = torch.tensor([b["adv"] for b in flat_dataset], device=device)
            std_advs, mean_advs = torch.std_mean(all_advs, unbiased=False)
            all_int_advs = torch.tensor(
                [b["adv_i"] for b in flat_dataset]
                )  
            std_int_advs, mean_int_advs = torch.std_mean(all_int_advs, unbiased=False)
        else:
            
        # puting limit in the times of each data
        time_recorder = {
            i:0 for i in range(len(dataset))
        }
        
        dataset_with_index = [
            (i, data) for i, data in enumerate(dataset)
            ]
    
        
        
        for _ in range(self.epochs):
            random.shuffle(dataset_with_index)
            for minibatch in _yield_subset_of_sequences_with_fixed_number_of_items_with_limit(
                dataset_with_index, 
                self.minibatch_size, 
                time_recorder
            ):
                
                self._update_once_recurrent(
                    minibatch, mean_advs, 
                    std_advs, std_int_advs, mean_int_advs)

        '''

        for _ in range(self.epochs):
            random.shuffle(dataset)
            for minibatch in _yield_subset_of_sequences_with_fixed_number_of_items(
                dataset, self.minibatch_size
            ):

                _, _, training_stop = self._update_once_recurrent(
                    minibatch, mean_advs, 
                    std_advs, std_int_advs, mean_int_advs)
                
                # early stopping
                if training_stop is True: 
                    break
                

    def _lossfun(
        self, entropy, vs_pred, log_probs, vs_pred_old, log_probs_old, advs, vs_teacher
    ):

        prob_ratio = torch.exp(log_probs - log_probs_old)

        loss_policy = -torch.mean(
            torch.min(
                prob_ratio * advs,
                torch.clamp(prob_ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advs,
            ),
        )

        if self.clip_eps_vf is None:
            loss_value_func = F.mse_loss(vs_pred, vs_teacher)
        else:
            clipped_vs_pred = _elementwise_clip(
                vs_pred,
                vs_pred_old - self.clip_eps_vf,
                vs_pred_old + self.clip_eps_vf,
            )
            loss_value_func = torch.mean(
                torch.max(
                    F.mse_loss(vs_pred, vs_teacher, reduction="none"),
                    F.mse_loss(clipped_vs_pred, vs_teacher, reduction="none"),
                )
            )
        
        loss_entropy = -torch.mean(entropy)

        self.value_loss_record.append(float(loss_value_func))
        self.policy_loss_record.append(float(loss_policy))
        

        loss = (
            loss_policy
            + self.value_func_coef * loss_value_func
            + self.entropy_coef * loss_entropy
        )

        return loss
    
    
    def update_lr(self):
        diff = self.lr_end - self.lr_start
        new_lr = self.lr_start + (
            diff * (
                self.n_updates/self.lr_decay_step
                )
            )
        for group in self.optimizer.param_groups:
            group['lr'] = new_lr
        self.curr_lr = new_lr
        
    def batch_act(self, batch_obs):
        if self.training:
            return self._batch_act_train(batch_obs)
        else:
            return self._batch_act_eval(batch_obs)

    def batch_observe(self,
     batch_obs, 
     batch_reward,
      batch_done, 
      batch_reset, batch_battle_result):
        if self.training:
            self._batch_observe_train(batch_obs, batch_reward, batch_done, batch_reset, batch_battle_result)
        else:
            self._batch_observe_eval(batch_obs, batch_reward, batch_done, batch_reset)

    def _batch_act_eval(self, batch_obs):
        assert not self.training
        b_state = self.batch_states(batch_obs, self.device, self.phi)

        if self.obs_normalizer:
            b_state = self.obs_normalizer(b_state)

        with torch.no_grad(), pfrl.utils.evaluating(self.model):
            if self.recurrent:
                if self.batch_last_action:
                    prev_action =torch.zeros(1, 14)
                    prev_action[:, self.batch_last_action[0]] = 1
                else:
                    prev_action = torch.zeros(1, 14)

                (action_prob, _, _), self.test_recurrent_states = one_step_forward(
                    self.model, b_state.to(self.device), prev_action.to(self.device), self.test_recurrent_states
                )
                action_distrib = pfrl.policies.SoftmaxCategoricalHead()(
                    torch.reshape(
                    action_prob,
                    (1, action_prob.shape[-1])
                )
                )
            else:
                action_prob, _ = self.model(b_state)
                action_distrib = pfrl.policies.SoftmaxCategoricalHead()(
                    torch.reshape(
                    action_prob,
                    (1, action_prob.shape[-1])
                )
                )
            if self.act_deterministically:
                action = mode_of_distribution(action_distrib).cpu().numpy()
            else:
                action = action_distrib.sample().cpu().numpy()
        
        self.batch_last_action = list(action)
        
        return action

    def _batch_act_train(self, batch_obs):
        assert self.training
        b_state = self.batch_states(batch_obs, self.device, self.phi)

        if self.obs_normalizer:
            b_state = self.obs_normalizer(b_state)

        num_envs = len(batch_obs)
        if self.batch_last_episode is None:
            self._initialize_batch_variables(num_envs)
        assert len(self.batch_last_episode) == num_envs
        assert len(self.batch_last_state) == num_envs
        assert len(self.batch_last_action) == num_envs

        # action_distrib will be recomputed when computing gradients
        with torch.no_grad(), pfrl.utils.evaluating(self.model):
            if self.recurrent:
                
                assert self.train_prev_recurrent_states is None
                
                if self.batch_last_action:
                    self.prev_action =torch.zeros(1,14)
                    self.prev_action[:,self.batch_last_action[0]] = 1
                else:
                    self.prev_action = torch.zeros(1, 14)
                
                self.train_prev_recurrent_states = self.train_recurrent_states
                (
                    (action_prob, batch_value, batch_int_value),
                    self.train_recurrent_states,
                ) = one_step_forward(
                    self.model, 
                    b_state.to(self.device), 
                    self.prev_action.to(self.device), 
                    self.train_prev_recurrent_states
                )
                
                action_distrib = pfrl.policies.SoftmaxCategoricalHead()(
                    torch.reshape(
                    action_prob,
                    (1, action_prob.shape[-1])
                )
                )
            
            else:
                action_distrib, batch_value = self.model(b_state)
            '''
            self.entropy_record.extend(action_distrib.entropy().cpu().numpy())
            self.value_record.extend(batch_value.cpu().numpy())
            self.int_value_records.extend(batch_int_value.cpu().numpy().item())
            
            '''
            batch_action = action_distrib.sample().cpu().numpy()
            self.entropy_record.append(action_distrib.entropy().cpu().numpy().item())
            self.value_record.append(batch_value.cpu().numpy().item())
            self.int_value_records.append(batch_int_value.cpu().numpy().item())
            
        self.batch_last_state = list(batch_obs)
        
        self.batch_last_last_action = self.batch_last_action
        self.batch_last_action = list(batch_action)
        
        
        
        
        return batch_action

    def _batch_observe_eval(self, batch_obs, batch_reward, batch_done, batch_reset):
        assert not self.training
        if self.recurrent:
            # Reset recurrent states when episodes end
            indices_that_ended = [
                i
                for i, (done, reset) in enumerate(zip(batch_done, batch_reset))
                if done or reset
            ]
            if indices_that_ended:
                self.test_recurrent_states = mask_recurrent_state_at(
                    self.test_recurrent_states, indices_that_ended
                )

    def _batch_observe_train(self, batch_obs, batch_reward, batch_done, batch_reset, batch_battle_result):
        assert self.training
        self.global_step += 1
        for i, (prev_action, state, action, reward, next_state, done, reset) in enumerate(
            zip(
                self.batch_last_last_action,
                self.batch_last_state,
                self.batch_last_action,
                batch_reward,
                batch_obs,
                batch_done,
                batch_reset,
            )
        ):
            if state is not None:
                assert action is not None
                
                transition = {
                    "prev_action": prev_action, 
                    "state": state,
                    "action": action,
                    "reward": reward,
                    "next_state": next_state,
                    "nonterminal": 0.0 if done else 1.0
                }
                

                # compute interinsic reward
                if self.rnd_model is not None:
                
                    transition["r_i"] = self.compute_intrinsic_reward(next_state, action)
                    self.int_reward_record.append(transition["r_i"])

                if self.recurrent:
                    transition["recurrent_state"] = get_recurrent_state_at(
                        self.train_prev_recurrent_states, i, detach=True
                    )
                    transition["next_recurrent_state"] = get_recurrent_state_at(
                        self.train_recurrent_states, i, detach=True
                    )
                    
                self.batch_last_episode[i].append(transition)
            if done or reset:
                assert self.batch_last_episode[i]
                self.len_episode_record.append(len(self.batch_last_episode[i]))
                self.step_per_episode += 1
                self.memory.append(self.batch_last_episode[i])
                self.batch_last_episode[i] = []
                
                self.batch_last_state[i] = None
                self.batch_last_action[i] = None
                self.batch_last_last_action[i] = 0

                
                

        self.train_prev_recurrent_states = None
        
        if self.recurrent:
            # Reset recurrent states when episodes end
            indices_that_ended = [
                i
                for i, (done, reset) in enumerate(zip(batch_done, batch_reset))
                if done or reset
            ]
            if indices_that_ended:
                self.train_recurrent_states = mask_recurrent_state_at(
                    self.train_recurrent_states, indices_that_ended
                )

    def compute_intrinsic_reward(self, next_obs, action):
       
        next_obs = torch.FloatTensor(next_obs).to(self.device)
        next_obs = torch.unsqueeze(next_obs, 0)

        
        action_onehot = torch.zeros((1, 14)).to(self.device)
        action_onehot[: , action] = 1
        with torch.no_grad():
        
            target_next_feature = self.rnd_model.target(next_obs, action_onehot)
            predict_next_feature = self.rnd_model.predictor(next_obs, action_onehot)
        intrinsic_reward = (target_next_feature - predict_next_feature).pow(2).mean(1)
        
        return intrinsic_reward.data.cpu().numpy().item()

    def save_rnd(self, path):
        assert self.rnd_model is not None
        torch.save(self.rnd_model.state_dict(), path)
    
    def load_rnd(self, path):
        assert self.rnd_model is not None
        self.rnd_model.load_state_dict(torch.load(path, map_location=self.device))
        
    def save_rnd_normalizer(self, path):
        assert self.rnd_normalizer is not None
        joblib.dump(self.rnd_normalizer, path, compress=3)
        
    def load_rnd_normalizer(self, path):
        assert self.rnd_normalizer is not None
        self.rnd_normalizer = joblib.load(path)
        
    def save_reward_normalizer(self, path):
        assert self.reward_normalizer is not None
        joblib.dump(self.reward_normalizer, path, compress=3)
        
    def load_reward_normalizer(self, path):
        assert self.reward_normalizer is not None
        self.reward_normalizer = joblib.load(path)
    
    def get_statistics(self):
        
        stats = [
            ("average_int_reward", _mean_or_nan(self.int_reward_record)),
            ("average_int_value", _mean_or_nan(self.int_value_records)),
            ("average_int_value_loss", _mean_or_nan(self.int_value_loss_record)),
            ("average_rnd_loss", _mean_or_nan(self.rnd_loss_record)),
            ("average_rnd_mean", _mean_or_nan(self.rnd_mean_record)),
            ("average_rnd_var", _mean_or_nan(self.rnd_var_record)),
            ("int_rewems", self.int_reward_fileter.rewems),
            ("average_approx_kl_div", float(_mean_or_nan(self.approx_kl_div_record))),
            ("average_value", _mean_or_nan(self.value_record)),
            ("average_entropy", _mean_or_nan(self.entropy_record)),
            ("average_value_loss", _mean_or_nan(self.value_loss_record)),
            ("average_policy_loss", _mean_or_nan(self.policy_loss_record)),
            ("average_length_episode", _mean_or_nan(self.len_episode_record)),
            ("n_updates", self.n_updates),
            ("explained_variance", self.explained_variance),
            ("int_explained_variance", self.int_explained_variance),
            ("entropy_coef", self.entropy_coef )
        ]
        if self.optimizer is not None:
            stats.append(
                ("curr_lr", self.curr_lr)
            )
        return stats
    




class DDA_PPO(PPO):
    def __init__(
        self, 
        model, 
        optimizer, 
        obs_normalizer: RunningMeanStd = None,
        reward_normalizer: RunningMeanStd = None,
        gpu=None, 
        gamma=0.99, 
        lambd=0.95, 
        phi=lambda x: x,
        value_func_coef=1, 
        entropy_coef=0.001, 
        update_interval=2048, 
        minibatch_size=64, 
        epochs=10, 
        clip_eps=0.2, 
        clip_eps_vf=None, 
        standardize_advantages=False, 
        batch_states=batch_states, 
        recurrent=False, 
        max_recurrent_sequence_len=None, 
        act_deterministically=False, 
        max_grad_norm=5, 
        value_stats_window=100, 
        entropy_stats_window=100, 
        value_loss_stats_window=100, 
        policy_loss_stats_window=100, 
        rnd_model=None, 
        rnd_normalizer=None, 
        memory: LocalMemory = None, 
        win_predictor: LSTM_Predictor = None,
        win_predictor_optim: torch.optim.Optimizer = None
        ):
        super().__init__(
            model, 
            optimizer, 
            obs_normalizer=obs_normalizer, 
            reward_normalizer=reward_normalizer,
            gpu=gpu, 
            gamma=gamma, 
            lambd=lambd, 
            phi=phi, 
            value_func_coef=value_func_coef, 
            entropy_coef=entropy_coef, 
            update_interval=update_interval, 
            minibatch_size=minibatch_size, 
            epochs=epochs, 
            clip_eps=clip_eps, 
            clip_eps_vf=clip_eps_vf, 
            standardize_advantages=standardize_advantages, 
            batch_states=batch_states, 
            recurrent=recurrent, 
            max_recurrent_sequence_len=max_recurrent_sequence_len, 
            act_deterministically=act_deterministically, 
            max_grad_norm=max_grad_norm, value_stats_window=value_stats_window, entropy_stats_window=entropy_stats_window, value_loss_stats_window=value_loss_stats_window, policy_loss_stats_window=policy_loss_stats_window, rnd_model=rnd_model, rnd_normalizer=rnd_normalizer, memory=memory)
        self.win_predictor = win_predictor
        self.win_predictor_optim = win_predictor_optim
        if gpu is not None:
            assert torch.cuda.is_available()
            if self.win_predictor is not None:
                self.win_predictor.to(self.device)
        
        self.win_predictor_loss_func= nn.BCELoss()
        record_size = 100
        self.dda_reward_record = collections.deque(maxlen=record_size)
        self.dda_loss_record = collections.deque(maxlen=record_size)
        self.dda_accuracy_record = collections.deque(maxlen=record_size)
        self.dda_alpha_record = collections.deque(maxlen=record_size)
        self.prev_loss = np.inf
        self.reward_record = collections.deque(maxlen=value_stats_window)
        self.final_reward_record = collections.deque(maxlen=record_size)
        # Recurrent states of the model
        self.dda_train_recurrent_states = None
        self.dda_train_prev_recurrent_states = None
        self.dda_test_recurrent_states = None
        self.dda_n_updates = 0

        if self.win_predictor_optim is not None:
            
            self.dda_lr_start = 5e-5
            self.dda_lr_decay_step = 5000000
            self.dda_lr_end = self.dda_lr_start/10
            self.dda_curr_lr = self.win_predictor_optim.defaults["lr"]

    def update_dda_lr(self):
        diff = self.dda_lr_end - self.dda_lr_start
        new_lr = self.dda_lr_start + (
            diff * (
                self.dda_n_updates/self.dda_lr_decay_step
                )
            )
        for group in self.win_predictor_optim.param_groups:
            group['lr'] = new_lr
        self.dda_curr_lr = new_lr
        
    def computing_dda_reward(self, state, reward):
        if self.win_predictor is not None:
            with torch.no_grad():
                self.dda_train_prev_recurrent_states = self.dda_train_recurrent_states
                state = torch.reshape(state, (1, 1, -1))
                prev_action = torch.reshape(self.prev_action, (1, 1, 14))
                (
                    pred_win_rate, 
                    self.dda_train_recurrent_states
                ) = self.win_predictor(
                    (state.to(self.device), prev_action.to(self.device)), 
                    self.dda_train_prev_recurrent_states
                    )
                pred_win_rate = torch.squeeze(pred_win_rate).cpu()
                dda_reward = 1 + (
                    0.5 - (
                    np.abs(pred_win_rate - 0.5)
                    ) 
                    )
                alpha = pred_win_rate
        self.dda_alpha_record.append([alpha])
        self.dda_reward_record.append(dda_reward)
        final_reward = (1-alpha)*reward + alpha*dda_reward
        #final_reward = reward + dda_reward
        
        return final_reward
    
    def sigmoid(self, a):
            return 1 / (1 + np.exp(-a))
        
    def _batch_observe_train(
        self, 
        batch_obs,
        batch_reward, 
        batch_done, 
        batch_reset, 
        batch_battle_result
    ):
        assert self.training
        self.global_step += 1
        for i, (prev_action, state, action, reward, next_state, done, reset) in enumerate(
            zip(
                self.batch_last_last_action,
                self.batch_last_state,
                self.batch_last_action,
                batch_reward,
                batch_obs,
                batch_done,
                batch_reset,
            )
        ):  

            if state is not None:
                assert action is not None
                self.reward_record.append(reward)
                # transforming reward to dda reward
                reward = self.computing_dda_reward(state, reward)
                
                self.final_reward_record.append(reward)
                transition = {
                    "prev_action": prev_action, 
                    "state": state,
                    "action": action,
                    "reward": reward,
                    "next_state": next_state,
                    "nonterminal": 0.0 if done else 1.0,
                    "battle_result": batch_battle_result
                }

                # compute interinsic reward
                if self.rnd_model is not None:
                
                    transition["r_i"] = self.compute_intrinsic_reward(next_state, action)
                    self.int_reward_record.append(transition["r_i"])
                if self.recurrent:
                    transition["recurrent_state"] = get_recurrent_state_at(
                        self.train_prev_recurrent_states, i, detach=True
                    )
                    transition["next_recurrent_state"] = get_recurrent_state_at(
                        self.train_recurrent_states, i, detach=True
                    )
                    transition["dda_recurrent_state"] = get_recurrent_state_at(
                        self.dda_train_prev_recurrent_states, i, detach=True
                    )
                    transition["dda_next_recurrent_state"] = get_recurrent_state_at(
                        self.dda_train_recurrent_states, i, detach=True
                    )
                self.batch_last_episode[i].append(transition)
            if done or reset:
                assert self.batch_last_episode[i]
                self.len_episode_record.append(len(self.batch_last_episode[i]))
                self.step_per_episode += 1
                self.memory.append(self.batch_last_episode[i])
                self.batch_last_episode[i] = []
                
                self.batch_last_state[i] = None
                self.batch_last_action[i] = None
                self.batch_last_last_action[i] = 0
                

        self.train_prev_recurrent_states = None
        self.dda_train_prev_recurrent_states = None
        if self.recurrent:
            # Reset recurrent states when episodes end
            indices_that_ended = [
                i
                for i, (done, reset) in enumerate(zip(batch_done, batch_reset))
                if done or reset
            ]
            if indices_that_ended:
                self.train_recurrent_states = mask_recurrent_state_at(
                    self.train_recurrent_states, indices_that_ended
                )
                
                self.dda_train_recurrent_states = mask_recurrent_state_at(
                    self.dda_train_recurrent_states, indices_that_ended
                )
    
    def _update_once_recurrent(
        self, episodes, mean_advs, std_advs, mean_int_advs, std_int_advs):

        assert std_advs is None or std_advs > 0

        device = self.device

        seqs_states, seqs_prev_actions, training_stop = super()._update_once_recurrent(
            episodes, mean_advs, std_advs, mean_int_advs, std_int_advs)

            
        # train win predictor
        if self.win_predictor is not None:
            with torch.no_grad(), pfrl.utils.evaluating(self.win_predictor):
                rs_h, rs_c = concatenate_recurrent_states(
                    [ep[0]["dda_recurrent_state"] for ep in episodes]
                )
                rs = tuple([rs_h.to(device), rs_c.to(device)])
                
    
            y = []
            for ep in episodes:
                label = ep[0]["battle_result"]
                y.append(label)

            y = torch.Tensor(y).float().to(device)

            x = (pad_sequence(seqs_states), pad_sequence(seqs_prev_actions))
            self.train_win_predictor(x, y, rs)
        self.update_dda_lr()

        return seqs_states, seqs_prev_actions, training_stop

    def train_win_predictor(self, x, y, rs):
   
        self.win_predictor_optim.zero_grad()
        pred, _ = self.win_predictor(x, rs)
        
        loss = self.win_predictor_loss_func(pred, y)
        loss.backward()
        self.win_predictor_optim.step()
        

        self.dda_loss_record.append(loss.detach().cpu().numpy().item())
        accuracy = self.binary_accuracy(y, pred)
        self.dda_accuracy_record.append(accuracy.detach().cpu().numpy())
        
        self.dda_n_updates+=1
        self.update_dda_lr()

    def binary_accuracy(self, y_true, y_pred):
        '''Calculates the mean accuracy rate across all predictions for binary
        classification problems.
        '''
        return torch.mean((y_true == torch.round(y_pred)).float())


            
        
    def save_win_predictor(self, path):
        assert self.win_predictor is not None
        torch.save(self.win_predictor.state_dict(), path)
    
    def load_win_predictor(self, path):
        assert self.win_predictor is not None
        self.win_predictor.load_state_dict(torch.load(path, map_location=self.device))
    
    def get_statistics(self):
        stats = super().get_statistics()
        if self.win_predictor_optim is not None:
            stats.append(
                ("dda_curr_lr", self.dda_curr_lr)
            )
            stats.append(
                ("dda_n_updates", self.dda_n_updates)
            )
        return stats

    def get_dda_statistics(self):

        return [
            ("average_dda_loss", _mean_or_nan(self.dda_loss_record)),
            ("average_dda_accuracy", _mean_or_nan(self.dda_accuracy_record)),
            ("average_dda_reward", _mean_or_nan(self.dda_reward_record)),
            ("average_dda_alpha", _mean_or_nan(self.dda_alpha_record)),
            ("average_reward", _mean_or_nan(self.reward_record)),
            ("average_final_reward", _mean_or_nan(self.final_reward_record))
        ]

            