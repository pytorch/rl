def bellman_max(next_observation, reward, done, gamma, value_model):
    qmax = value_model(next_observation).max(dim=-1)[0]
    nonterminal_target = reward + gamma * qmax
    terminal_target = reward
    target = done * terminal_target + (~done) * nonterminal_target
    return target
