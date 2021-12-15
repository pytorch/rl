import torch

from torchrl.objectives.returns.vtrace import vtrace


class QValEstimator:
    def __init__(self, value_model):
        self.value_model = value_model

    @property
    def device(self):
        return next(self.value_model.parameters()).device

    def __call__(self, tensordict):
        tensordict_device = tensordict.to(self.device)
        self.value_model_device(tensordict_device) # udpates the value key
        gamma = tensordict_device.get("gamma")
        reward = tensordict_device.get("reward")
        next_value = torch.cat([tensordict_device.get("value")[:, 1:], torch.ones_like(reward[:, :1])], 1)
        q_value = reward + gamma * next_value
        tensordict_device.set("q_value", q_value)

class VTraceEstimator:
    def __call__(self, tensordict):
        tensordict_device = tensordict.to(device)
        rewards = tensordict_device.get("reward")
        vals = tensordict_device.get("value")
        log_mu = tensordict_device.get("log_mu")
        log_pi = tensordict_device.get("log_pi")
        gamma = tensordict_device.get("gamma")
        v_trace, rho = vtrace(rewards, vals, log_pi, log_mu, gamma, rho_bar=self.rho_bar, c_bar=self.c_bar)
        tensordict_device.set("v_trace", v_trace)
        tensordict_device.set("rho", rho)
        return tensordict_device

class ImpalaLoss:
    def __call__(self, tensordict):
        tensordict_device = tensordict.to(device)
        self.q_val_estimator(tensordict_device)
        self.v_trace_estimator(tensordict_device)