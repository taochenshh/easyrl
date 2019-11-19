from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Dict
from typing import List


@dataclass
class StepData:
    ob: Any = None
    state: Any = None
    action: Any = None
    reward: float = None
    done: bool = None
    info: Dict = None
    # whether the time step reaches the episode limit
    timeout: bool = False

    def __post_init__(self):
        self.done = self.done and not self.timeout


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
        return [step_data.ob for step_data in self.traj_data]

    @property
    def states(self):
        return [step_data.state for step_data in self.traj_data]

    @property
    def actions(self):
        return [step_data.action for step_data in self.traj_data]

    @property
    def rewards(self):
        return [step_data.reward for step_data in self.traj_data]

    @property
    def dones(self):
        return [step_data.done for step_data in self.traj_data]

    @property
    def infos(self):
        return [step_data.info for step_data in self.traj_data]

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
