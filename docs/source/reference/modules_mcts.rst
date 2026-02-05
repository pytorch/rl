.. currentmodule:: torchrl.modules.mcts

torchrl.modules.mcts package
============================

This module provides Monte Carlo Tree Search (MCTS) components, including score computation modules for balancing exploration and exploitation in tree search algorithms.

MCTS Policies
-------------

.. autosummary::
   :toctree: generated/

   MCTSPolicyBase
   MCTSPolicy
   AlphaGoPolicy
   AlphaStarPolicy
   MuZeroPolicy

MCTS Scores
-----------

.. autosummary::
   :toctree: generated/

   MCTSScore
   PUCTScore
   UCBScore
   EXP3Score
   UCB1TunedScore
   MCTSScores
