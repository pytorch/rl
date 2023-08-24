from pettingzoo.classic import tictactoe_v3

if __name__ == "__main__":
    env = tictactoe_v3.env(render_mode="human")
    env.reset(seed=42)

    for _ in range(5):
        agent = env.agent_selection

        print(f"\nStepping agent {env.agent_selection}")
        print(f"current agents {env.agents}, possible ones {env.possible_agents}")

        action = env.action_space(
            agent
        ).sample()  # this is where you would insert your policy

        env.step(action)

        erminations_dict = env.terminations
        truncations_dict = env.truncations
        info_dict = env.infos
        rewards_dict = env.rewards
        observation_dict = {agent: env.observe(agent) for agent in env.possible_agents}
        print(f"obs {observation_dict}, don {truncations_dict} {erminations_dict}")
    env.close()
