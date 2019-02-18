"""
This should results in an average return of ~3000 by the end of training.

Usually hits 3000 around epoch 80-100. Within a see, the performance will be
a bit noisy from one epoch to the next (occasionally dips dow to ~2000).

Note that one epoch = 5k steps, so 200 epochs = 1 million steps.
"""
import gym
import os

os.environ['LD_LIBRARY_PATH']+=':'+os.environ['HOME']+'/.mujoco/mjpro150/bin:'

import sys
sys.path.append('../../')

import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.obs_dict_replay_buffer import ObsDictRelabelingBuffer
from rlkit.exploration_strategies.base import (
    PolicyWrappedWithExplorationStrategy
)
from rlkit.exploration_strategies.gaussian_and_epsilon_strategy import (
    GaussianAndEpislonStrategy
)
from rlkit.launchers.launcher_util import setup_logger
from rlkit.torch.her.her import HerTd3, HerTwinSAC
from rlkit.torch.networks import FlattenMlp, TanhMlpPolicy
from rlkit.torch.sac.policies import TanhGaussianPolicy



def experiment(variant):
    env = gym.make('FetchPush-v1')
    es = GaussianAndEpislonStrategy(
        action_space=env.action_space,
        max_sigma=.2,
        min_sigma=.2,  # constant sigma
        epsilon=.3,
    )
    obs_dim = env.observation_space.spaces['observation'].low.size
    goal_dim = env.observation_space.spaces['desired_goal'].low.size
    action_dim = env.action_space.low.size
    qf1 = FlattenMlp(
        input_size=obs_dim + goal_dim + action_dim,
        output_size=1,
        hidden_sizes=[400, 300],
    )
    qf2 = FlattenMlp(
        input_size=obs_dim + goal_dim + action_dim,
        output_size=1,
        hidden_sizes=[400, 300],
    )
    vf = FlattenMlp(
        hidden_sizes=[400, 300],
        input_size=obs_dim + goal_dim,
        output_size=1,
    )
    policy = TanhGaussianPolicy(
        hidden_sizes=[400, 300],
        obs_dim=obs_dim + goal_dim,
        action_dim=action_dim,
    )
    exploration_policy = PolicyWrappedWithExplorationStrategy(
        exploration_strategy=es,
        policy=policy,
    )
    replay_buffer = ObsDictRelabelingBuffer(
        env=env,
        **variant['replay_buffer_kwargs']
    )
    algorithm = HerTwinSAC(
        her_kwargs=dict(
            observation_key='observation',
            desired_goal_key='desired_goal'
        ),
        tsac_kwargs = dict(
            env=env,
            policy=policy,
            qf1=qf1,
            qf2=qf2,
            vf=vf,
            soft_target_tau=0.001,
            policy_lr=3E-4,
            qf_lr=3E-4,
            vf_lr=3E-4
        ),
        replay_buffer=replay_buffer,
        **variant['algo_kwargs']
    )
    algorithm.to(ptu.device)
    algorithm.train()


if __name__ == "__main__":
    variant = dict(
        algo_kwargs=dict(
            num_epochs=1000,
            num_steps_per_epoch=5000,
            num_steps_per_eval=1000,
            max_path_length=50,
            batch_size=128,
            discount=0.99,
        ),
        replay_buffer_kwargs=dict(
            max_size=100000,
            fraction_goals_rollout_goals=0.2,  # equal to k = 4 in HER paper
            fraction_goals_env_goals=0.0,
        ),
    )
    setup_logger('her-tsac-push-experiment', variant=variant)
    experiment(variant)
