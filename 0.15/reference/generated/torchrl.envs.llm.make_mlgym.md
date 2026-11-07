# make_mlgym

*class*torchrl.envs.llm.make_mlgym(***, *task: Literal['prisonersDilemma'] | None = None*, *tasks: list[Literal['prisonersDilemma']] | None = None*, *tokenizer: transformers.AutoTokenizer | str | None = None*, *device='cpu'*, *reward_wrong_format: float | None = None*)[[source]](../../_modules/torchrl/envs/llm/libs/mlgym.html#make_mlgym)

Wraps an MLGymEnv in a TorchRL Environment.

The appended transforms will make sure that the data is formatted for the LLM during (for the outputs of env.step)
and for the MLGym API (for inputs to env.step).

Keyword Arguments:

- **task** (*str*) -

The task to wrap. Exclusive with tasks argument.

Note

The correct format is simply the task name, e.g., "prisonersDilemma".
- **tasks** (*List**[**str**]*) -

The tasks available for the env. Exclusive with task argument.

Note

The correct format is simply the task name, e.g., "prisonersDilemma".
- **tokenizer** (*transformers.AutoTokenizer**or**str**,**optional*) - A transformer that tokenizes the data.
If a string is passed, it will be converted to a transformers.AutoTokenizer.
- **device** (*str**,**optional*) - The device to set to the env. Defaults to "cpu".
- **reward_wrong_format** ([*float*](torchrl.data.llm.TopKRewardSelector.html#torchrl.data.llm.TopKRewardSelector.float)*,**optional*) - The reward (negative penalty) for wrongly formatted actions.
Defaults to None (no penalty).