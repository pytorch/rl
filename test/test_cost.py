from torchrl.data.batchers.batcher import MultiStep
from torchrl.data.batchers.utils import expand_as_right
from torchrl.modules.distributions.continuous import Delta
from torchrl.data.tensordict.tensordict import assert_allclose_td
from torchrl.modules import DistributionalQValueActor, QValueActor
from torchrl.modules.models.models import MLP
from torchrl.data import TensorDict
from torch import nn
import torch
import pytest


def get_devices():
    devices = [torch.device("cpu")]
    for i in range(torch.cuda.device_count()):
        device += [torch.device(f"cuda:{i}")]
    return devices

class TestDQN:
    seed = 0


    def _create_mock_actor(self, batch=2, obs_dim=3, action_dim=4):
        # Actor
        mapping_operator = nn.Linear(obs_dim, action_dim)
        actor = QValueActor(mapping_operator, distribution_class=Delta)
        return actor

    def _create_mock_distributional_actor(self, batch=2, obs_dim=3, action_dim=4, atoms=5, vmin=1, vmax=5):
        # Actor
        support = torch.linspace(vmin, vmax, atoms, dtype=torch.float)
        mapping_operator = MLP(obs_dim, (atoms, action_dim))
        actor = DistributionalQValueActor(mapping_operator, support=support, distribution_class=Delta)
        return actor

    def _create_mock_data_dqn(self, batch=2, obs_dim=3, action_dim=4, atoms=None):
        #  create a tensordict
        obs = torch.randn(batch, obs_dim)
        next_obs = torch.randn(batch, obs_dim)
        if atoms:
            action_value = torch.randn(batch, atoms, action_dim).softmax(-2)
            action = (action_value[..., 0, :] == action_value[..., 0, :].max(-1, True)[0]).to(torch.long)
        else:
            action_value = torch.randn(batch, action_dim)
            action = (action_value == action_value.max(-1, True)[0]).to(torch.long)
        reward = torch.randn(batch, 1)
        done = torch.zeros(batch, 1, dtype=torch.bool)
        td = TensorDict(
            batch_size=(batch,),
            source={
                "observation": obs,
                "next_observation": next_obs,
                "done": done,
                "reward": reward,
                "action": action,
                "action_value": action_value,
            },
        )
        return td

    def _create_seq_mock_data_dqn(self, batch=2, T=4, obs_dim=3, action_dim=4, atoms=None):
        #  create a tensordict
        total_obs = torch.randn(batch, T+1, obs_dim)
        obs = total_obs[:, :T]
        next_obs = total_obs[:, 1:]
        if atoms:
            action_value = torch.randn(batch, T, atoms, action_dim).softmax(-2)  
            action = (action_value[..., 0,:] == action_value[..., 0,:].max(-1, True)[0]).to(torch.long)
        else:
            action_value = torch.randn(batch, T, action_dim)
            action = (action_value == action_value.max(-1, True)[0]).to(torch.long)
        reward = torch.randn(batch, T, 1)
        done = torch.zeros(batch, T, 1, dtype=torch.bool)
        mask = ~torch.zeros(batch, T, 1, dtype=torch.bool)
        td = TensorDict(
            batch_size=(batch, T),
            source={
                "observation": obs*mask.to(obs.dtype),
                "next_observation": next_obs*mask.to(obs.dtype),
                "done": done,
                "mask": mask,
                "reward": reward*mask.to(obs.dtype),
                "action": action*mask.to(obs.dtype),
                "action_value": action_value*expand_as_right(mask.to(obs.dtype).squeeze(-1), action_value),
            },
        )
        return td

    @pytest.mark.parametrize("loss_class", (DQNLoss, DoubleDQNLoss))
    def test_dqn(self, loss_class):
        torch.manual_seed(self.seed)
        actor = self._create_mock_actor()
        td = self._create_mock_data_dqn()
        loss_fn = loss_class(actor, gamma=0.9)
        loss = loss_fn(td)
        loss.backward()
        assert torch.nn.utils.clip_grad.clip_grad_norm_(actor.parameters(), 1.0) > 0.0

    @pytest.mark.parametrize("n", range(4))
    @pytest.mark.parametrize("loss_class", (DQNLoss, DoubleDQNLoss))    
    def test_dqn_batcher(self, n, loss_class, gamma=0.9):
        torch.manual_seed(self.seed)
        actor = self._create_mock_actor()
        
        td = self._create_seq_mock_data_dqn()
        loss_fn = loss_class(actor, gamma=gamma)

        ms = MultiStep(gamma=gamma, n_steps_max=n)
        ms_td = ms(td.clone())
        loss_ms = loss_fn(ms_td)
        with torch.no_grad():
            loss = loss_fn(td)
        if n == 0:
            assert_allclose_td(td, ms_td.select(*list(td.keys())))
            assert abs(loss-loss_ms)<1e-3, f"found abs(loss-loss_ms) = {abs(loss-loss_ms):4.5f} for n=0"
        else:
            with pytest.raises(AssertionError):
                torch.testing.assert_allclose(loss, loss_ms)
        loss_ms.backward()
        assert torch.nn.utils.clip_grad.clip_grad_norm_(actor.parameters(), 1.0) > 0.0


    @pytest.mark.parametrize("atoms", range(4, 10))
    @pytest.mark.parametrize("loss_class", (DistributionalDQNLoss, DistributionalDoubleDQNLoss))    
    @pytest.mark.parametrize("device", get_devices())    
    def test_distributional_dqn(self, atoms, loss_class, device, gamma=0.9):
        torch.manual_seed(self.seed)
        actor = self._create_mock_distributional_actor(atoms=atoms).to(device)
        
        td = self._create_mock_data_dqn(atoms=atoms).to(device)
        loss_fn = loss_class(actor, gamma=gamma)

        loss = loss_fn(td)
        loss.backward()
        assert torch.nn.utils.clip_grad.clip_grad_norm_(actor.parameters(), 1.0) > 0.0

if __name__ == "__main__":
    pytest.main([__file__])
