#  Copyright (c) 2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.
from torchrl.envs.libs.meltingpot import MeltingpotEnv

if __name__ == "__main__":
    from meltingpot import substrate as mp_substrate

    substrate_config = mp_substrate.get_config("commons_harvest__open")
    env_torchrl = MeltingpotEnv(substrate_config)
    td_reset = env_torchrl.reset()
    td_in = env_torchrl.rand_action(td_reset.clone())
    td = env_torchrl.step(td_in.clone())
