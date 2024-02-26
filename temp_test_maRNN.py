from torchrl.modules import MultiAgentLSTM, MultiAgentMLP
import torch

n_agents = 3
rnn_input_size = 10
rnn_hidden_size = 16
batch_size = 32
dummy_obs = torch.zeros(batch_size, n_agents, rnn_input_size)

rnn = MultiAgentLSTM(
    n_agents=n_agents,
    centralised=False,
    share_params=True,
    out_features=rnn_hidden_size,
    mlp_kwargs={"out_features": rnn_input_size},
    lstm_kwargs={"input_size": rnn_input_size, "hidden_size": rnn_hidden_size},
)
print(rnn)
print(rnn(dummy_obs))

# mlp = MultiAgentMLP(
#     n_agent_inputs=n_agent_inputs,
#     n_agent_outputs=n_agent_outputs,
#     n_agents=n_agents,
#     centralised=False,
#     share_params=True,
#     depth=2,
# )
# print(mlp)
# TODO: sequential these two
