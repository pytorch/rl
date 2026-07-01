# torchrl.modules.mcts package

This module provides Monte Carlo Tree Search (MCTS) components, including score computation modules for balancing exploration and exploitation in tree search algorithms.

## MCTS Scores

| [`MCTSScore`](generated/torchrl.modules.mcts.MCTSScore.html#torchrl.modules.mcts.MCTSScore)(*args, **kwargs) | Abstract base class for MCTS score computation modules. |
| --- | --- |
| [`PUCTScore`](generated/torchrl.modules.mcts.PUCTScore.html#torchrl.modules.mcts.PUCTScore)(*args, **kwargs) | Computes the PUCT (Polynomial Upper Confidence Trees) score for MCTS. |
| [`UCBScore`](generated/torchrl.modules.mcts.UCBScore.html#torchrl.modules.mcts.UCBScore)(*args, **kwargs) | Computes the UCB (Upper Confidence Bound) score, specifically UCB1, for MCTS. |
| [`EXP3Score`](generated/torchrl.modules.mcts.EXP3Score.html#torchrl.modules.mcts.EXP3Score)(*args, **kwargs) | Computes action selection probabilities for the EXP3 algorithm in MCTS. |
| [`UCB1TunedScore`](generated/torchrl.modules.mcts.UCB1TunedScore.html#torchrl.modules.mcts.UCB1TunedScore)(*args, **kwargs) | Computes the UCB1-Tuned score for MCTS, using variance estimation. |
| [`MCTSScores`](generated/torchrl.modules.mcts.MCTSScores.html#torchrl.modules.mcts.MCTSScores)(value[, names, module, qualname, ...]) | A collection of MCTS score computation modules. |