import importlib
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torchrl.data import (
    CompositeSpec,
    DiscreteTensorSpec,
    TensorSpec,
    UnboundedContinuousTensorSpec,
)
from torchrl.envs import GymLikeEnv

_has_open_spiel = importlib.util.find_spec("open_spiel") is not None

if _has_open_spiel:
    import open_spiel


class OpenSpielEnv(GymLikeEnv):
    """OpenSpiel environment wrapper.

    The OpenSpiel can be found here: https://github.com/google-deepmind/open_spiel/.

    Paper: https://arxiv.org/abs/1908.09453

    """

    git_url = "https://github.com/google-deepmind/open_spiel/"
    libname = "open_spiel"

    def __init__(self, env=None, **kwargs):
        if env is not None:
            kwargs["env"] = env
        super().__init__(**kwargs)
        self.state: Optional[open_spiel.python.rl_environment.TimeStep] = None

    @property
    def lib(self):
        import open_spiel

        return open_spiel

    def _output_transform(
        self, step_output: "open_spiel.python.rl_environment.TimeStep"
    ) -> Tuple[
        Any,
        float | np.ndarray,
        bool | np.ndarray | None,
        bool | np.ndarray | None,
        bool | np.ndarray | None,
        dict,
    ]:
        self.state = step_output

        rewards = np.asarray(step_output.rewards)
        obs = self._get_observation()
        done = step_output.step_type == open_spiel.python.rl_environment.StepType.LAST

        return (
            obs,
            rewards,
            done,
            done,
            done,
            None,
        )

    def _reset_output_transform(self, step_output: Tuple) -> Tuple:
        self.state = step_output
        return self._get_observation(), None

    def _get_observation(self):
        state = self.state.observations["info_state"]
        return torch.Tensor(
            state,
            device=self.device,
        )

    def read_action(self, action):
        action_np = super().read_action(action)
        return action_np[
            self.state.current_player() : self.state.current_player() + 1, ...
        ]

    def _check_kwargs(self, kwargs: Dict):
        pass

    def _init_env(self) -> Optional[int]:
        pass

    def _build_env(self, **kwargs) -> "open_spiel.python.rl_environment.Environment":
        return open_spiel.python.rl_environment.Environment(**kwargs)

    def _make_specs(self, env: "open_spiel.python.rl_environment.Environment") -> None:
        spec = env.observation_spec()
        num_players = env.num_players

        self.observation_spec = self._make_observation_spec(
            spec["info_state"], num_players
        )
        self.reward_spec = self._make_reward_spec(num_players)
        self.action_spec = self._make_action_spec(env.action_spec(), num_players)
        self.done_spec = self.done_spec = DiscreteTensorSpec(
            n=2,
            shape=torch.Size((1,)),
            dtype=torch.bool,
        )

    def _make_observation_spec(
        self, info_state: Tuple[int, ...], num_players: int
    ) -> TensorSpec:
        return CompositeSpec(
            {
                "observation": UnboundedContinuousTensorSpec(
                    shape=(num_players,) + info_state,
                    device=self.device,
                )
            },
            shape=(),
        )

    def _make_reward_spec(self, num_players: int):
        return CompositeSpec(
            {
                "reward": UnboundedContinuousTensorSpec(
                    shape=(num_players,),
                    device=self.device,
                )
            },
            shape=(),
        )

    def _make_action_spec(self, org_action_spec: Dict[str, int], num_players: int):
        dtype = org_action_spec["dtype"]
        if dtype is not int:
            raise ValueError(f"{dtype} is not supported yet")
        return CompositeSpec(
            {
                "action": DiscreteTensorSpec(
                    n=org_action_spec["num_actions"],
                    shape=(num_players,),
                    device=self.device,
                )
            },
            shape=(),
        )

    def _set_seed(self, seed: Optional[int]):
        if seed is not None:
            self._env.seed(seed)
