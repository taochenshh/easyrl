from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Dict
from typing import List

import numpy as np


@dataclass
class StepData:
    ob: Any = None
    state: Any = None
    action: Any = None
    # store action infomation such as log probability, entropy
    action_info: Dict = None
    next_ob: Any = None
    next_state: Any = None
    reward: float = None
    done: bool = None
    info: Dict = None

    def __post_init__(self):
        """
        If the ob is a dict containing keys: ob and state
        then store them into ob and state separately
        """
        if isinstance(self.ob, dict):
            self.ob, self.state = self.dict_ob_state(self.ob)
        if isinstance(self.next_ob, dict):
            self.next_ob, self.next_state = self.dict_ob_state(self.next_ob)

    def dict_ob_state(self, ob):
        keys = ['ob', 'state']
        for key in keys:
            if key not in ob:
                raise ValueError('ob must have `ob` and `state` '
                                 'as keys if it is a dict!')
        state = ob['state']
        ob = ob['ob']
        return ob, state


@dataclass
class Trajectory:
    traj_data: List[StepData] = field(default_factory=list)

    def __len__(self):
        return len(self.traj_data)

    def __getitem__(self, item):
        return self.traj_data[item]

    def add(self, step_data=None, **kwargs):
        if step_data is not None:
            if not isinstance(step_data, StepData):
                raise TypeError('step_data should be an '
                                'instance of StepData!')
        else:
            step_data = StepData(**kwargs)
        self.traj_data.append(step_data)

    @property
    def obs(self):
        return np.array([step_data.ob for step_data in self.traj_data])

    @property
    def states(self):
        return np.array([step_data.state for step_data in self.traj_data])

    @property
    def actions(self):
        return np.array([step_data.action for step_data in self.traj_data])

    @property
    def actions_info(self):
        return np.array([step_data.action_info for step_data in self.traj_data])

    @property
    def next_obs(self):
        return np.array([step_data.next_ob for step_data in self.traj_data])

    @property
    def next_states(self):
        return np.array([step_data.next_state for step_data in self.traj_data])

    @property
    def rewards(self):
        return np.array([step_data.reward for step_data in self.traj_data])

    @property
    def dones(self):
        return np.array([step_data.done for step_data in self.traj_data])

    @property
    def infos(self):
        return [step_data.info for step_data in self.traj_data]

    @property
    def total_steps(self):
        return self.traj_data[0].action.shape[0] * len(self.traj_data)

    @property
    def steps_til_done(self):
        steps = []
        for i in range(self.dones.shape[1]):
            dones = self.dones[:, i]
            if not np.any(dones):
                steps.append(len(dones))
            else:
                steps.append(np.argmax(self.dones[:, i]) + 1)
        return np.array(steps)

    @property
    def episode_returns(self):
        """
        return the undiscounted return in each episode (between any two dones)

        Returns:
            list: a list of length-num_envs,
            each element in this list is a list of episodic return values

        """
        all_epr = []
        for i in range(self.dones.shape[1]):
            epr = []
            dones = self.dones[:, i]
            if not np.any(dones):
                epr.append(np.sum(self.rewards[:, i]))
            else:
                done_idx = np.where(dones)[0]
                t = 0
                for idx in done_idx:
                    epr.append(np.sum(self.rewards[t: idx + 1, i]))
                    t = idx + 1
            all_epr.append(epr)
        return all_epr

    def pop(self):
        """
        Remove and return the last element from the trajectory
        """
        step_data = self.traj_data.pop(-1)
        return step_data

    def popleft(self):
        """
        Remove and return the first element from the trajectory
        """
        step_data = self.traj_data.pop(0)
        return step_data
