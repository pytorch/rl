# make_gsm8k_env

*class*torchrl.envs.llm.make_gsm8k_env(*dataset: str = 'openai/gsm8k'*, *num_envs: int = 1*, *repeats: int | None = None*, *batch_size_dl: int = 1*, *seed: int | None = None*, *group_repeats: bool = False*, *tokenizer: transformers.PretrainedTokenizer | None = None*)[[source]](../../_modules/torchrl/envs/llm/datasets/gsm8k.html#make_gsm8k_env)

A builder for an LLMEnv-based GSM8K environment.

Note

Prefer torchrl.envs.llm.GSM8KEnv to interact with this dataset.