# torchrl.modules.mcts.MCTSScores

*class*torchrl.modules.mcts.MCTSScores(*value*, *names=None*, ***, *module=None*, *qualname=None*, *type=None*, *start=1*, *boundary=None*)[[source]](../../_modules/torchrl/modules/mcts/scores.html#MCTSScores)

A collection of MCTS score computation modules.

This enum provides a convenient way to create instances of the different MCTS score computation modules.
Each member of the enum is a callable that returns an instance of the corresponding score computation module.

Content:

- PUCTScore: Computes the PUCT score for MCTS.
- UCBScore: Computes the UCB score for MCTS.
- UCB1TunedScore: Computes the UCB1-Tuned score for MCTS.
- EXP3Score: Computes the EXP3 score for MCTS.

__init__(**args*, ***kwds*)

Attributes

| `PUCT` | |
| --- | --- |
| `UCB` | |
| `UCB1_TUNED` | |
| `EXP3` | |