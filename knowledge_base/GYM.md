# Working with gym

## What is OpenAI Gym?

OpenAI Gym is a python library that provides the tooling for coding and using
environments in RL contexts. The environments can be either simulators or real
world systems (such as robots or games).
Due to its easiness of use, Gym has been widely adopted as one the main APIs for
environment interaction in RL and control. 

Historically, Gym was started by OpenAI on [https://github.com/openai/gym](https://github.com/openai/gym).
Since then, OpenAI has ceased to maintain it and the library has been forked out
in [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) by the Farama Foundation.

Check the [Gym documentation](https://www.gymlibrary.dev/) for further details
about the installation and usage.

## Versioning
The OpenAI Gym library is known to have gone through multiple BC breaking changes
and significant user-facing API modifications.
In practice, TorchRL is tested against gym 0.13 and further and should work with
any version in between.

However, libraries built around  Gym may have a custom env construction process
that breaks the automatic wrapping from the `GymEnv` class. In those cases, it
is best to first create the gym environment and wrap it using
`torchrl.envs.libs.gym.GymWrapper`.

If you run into an issue when running TorchRL with a specific version of gym, 
feel free to open an issue and we will gladly look into this.
