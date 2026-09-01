# LLM Environments

The environment layer orchestrates data loading, tool execution, reward computation, and formatting.

| [`ChatEnv`](generated/torchrl.envs.llm.ChatEnv.html#torchrl.envs.llm.ChatEnv)(*args[, with_tokenizer]) | A chat-based environment for LLMs, designed as a blank canvas for conversation and RL. |
| --- | --- |
| [`CountdownEnv`](generated/torchrl.envs.llm.CountdownEnv.html#torchrl.envs.llm.CountdownEnv)(*args, **kwargs) | Countdown numbers-game environment for LLM post-training. |
| [`CountdownRewardParser`](generated/torchrl.envs.llm.CountdownRewardParser.html#torchrl.envs.llm.CountdownRewardParser)([tokenizer, in_keys, ...]) | Reward parser for the Countdown numbers game. |
| [`DatasetChatEnv`](generated/torchrl.envs.llm.DatasetChatEnv.html#torchrl.envs.llm.DatasetChatEnv)(*args, **kwargs) | Base class for chat environment with queries pulled from a dataset. |
| [`GSM8KEnv`](generated/torchrl.envs.llm.GSM8KEnv.html#torchrl.envs.llm.GSM8KEnv)(*args, **kwargs) | GSM8K dataset environment. |
| [`make_gsm8k_env`](generated/torchrl.envs.llm.make_gsm8k_env.html#torchrl.envs.llm.make_gsm8k_env)([dataset, num_envs, repeats, ...]) | A builder for an LLMEnv-based GSM8K environment. |
| [`GSM8KPrepareQuestion`](generated/torchrl.envs.llm.GSM8KPrepareQuestion.html#torchrl.envs.llm.GSM8KPrepareQuestion)([in_keys, out_keys]) | A transform to prepare the prompt when using GSM8k within an LLMEnv. |
| [`GSM8KRewardParser`](generated/torchrl.envs.llm.GSM8KRewardParser.html#torchrl.envs.llm.GSM8KRewardParser)([tokenizer, in_keys, ...]) | Reward parser for GSM8KEnv or make_gsm8k_env. |
| [`IFEvalEnv`](generated/torchrl.envs.llm.IFEvalEnv.html#torchrl.envs.llm.IFEvalEnv)(*args, **kwargs) | A chat environment based on the IFEval dataset. |
| [`IfEvalScorer`](generated/torchrl.envs.llm.IfEvalScorer.html#torchrl.envs.llm.IfEvalScorer)(*[, instruction_ids_key, ...]) | Scorer for the IF-Eval task. |
| [`IFEvalScoreData`](generated/torchrl.envs.llm.IFEvalScoreData.html#torchrl.envs.llm.IFEvalScoreData)(prompt_level_strict_acc, ...) | |
| [`MATHEnv`](generated/torchrl.envs.llm.MATHEnv.html#torchrl.envs.llm.MATHEnv)(*args, **kwargs) | MATH (competition mathematics) dataset environment. |
| [`MATHRewardParser`](generated/torchrl.envs.llm.MATHRewardParser.html#torchrl.envs.llm.MATHRewardParser)([tokenizer, in_keys, ...]) | Reward parser for the MATH (competition mathematics) dataset. |
| [`LLMEnv`](generated/torchrl.envs.llm.LLMEnv.html#torchrl.envs.llm.LLMEnv)(*args, **kwargs) | A text generation environment for language models. |
| [`LLMHashingEnv`](generated/torchrl.envs.llm.LLMHashingEnv.html#torchrl.envs.llm.LLMHashingEnv)(*args, **kwargs) | A text generation environment that uses a hashing module to identify unique observations. |
| [`make_mlgym`](generated/torchrl.envs.llm.make_mlgym.html#torchrl.envs.llm.make_mlgym)(*[, task, tasks, tokenizer, ...]) | Wraps an MLGymEnv in a TorchRL Environment. |
| [`MLGymWrapper`](generated/torchrl.envs.llm.MLGymWrapper.html#torchrl.envs.llm.MLGymWrapper)(*args, **kwargs) | A thin wrapper for MLGym environments. |