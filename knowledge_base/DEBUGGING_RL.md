# Things to consider when debugging RL

## General

### Have you validated your algorithm implementation on a few small, toy problems with known optimal returns e.g. gridworlds, mountaincar?
* Reason: This will reveal any extreme bugs in your implementation.
### Have you visualized your agents?
* Reason: This will reveal things the learning curves won’t tell you (i.e., bug or exploit in a video game).
### Be very careful with any data augmentation.
* Reason: Data augmentation cannot be applied to RL in the same ways as CV since an agent needs to act based on the observation. As an example, flipping an image may correspondingly “flip” the appropriate action.

## Policy

### Does the entropy of your policy converge too quickly, too slowly or change drastically?
* Reason: This can be algorithm dependent, but the entropy of the policy is roughly inversely related to the expected value of actions.
* Prescription: Tuning the coefficient of an entropy bonus (i.e., beta in PPO) can help entropies that converge too quickly/slowly. Alternatively, reducing/increasing the magnitude of rewards may also help if converging too quickly/slowly. Entropy curves that step-change dramatically are usually downstream of an issue with the problem formulation (i.e., obs or action space), learning rate, gradient norms or a bug in the implementation. 

## Rewards (beyond “going up”)

### Is the agent favoring a single component of the reward function (i.e. velocity vs L2 action magnitude)? 
* Reason: It may be the case that one of the components of the reward function is “easier” to optimize and so an agent will find the behavior as a local optima. 
* Prescription: In addition to tuning coefficients of reward components, it may also make sense to use the product of components instead of the sum. Tracking the stats w.r.t. each reward component may also yield insight. Alternatively, if some components are considered ‘auxiliary’, decaying the weight over time may be helpful.
### Is the task horizon extremely long?
* Reason: Credit assignment (i.e., attributing future/value rewards to past state/actions) becomes more difficult with the time between action and corresponding reward. In sparse reward environments, this can be a source of training inefficiency requiring many interactions with the environment.
* Prescription: Adding intermediate rewards for behaviors that are instrumental to the final goal can greatly increase training speed (e.g., in a soccer environment, an intermediate reward for kicking the ball will increase the likelihood that an agent discovers scoring a goal is rewarding).  This may create undesired optima though as exploiting the intermediate reward may unintentionally be more valuable than the true reward or lead to undesired idiosyncratic behaviors. One can decay the value of this intermediate reward to zero using a step or reward based curriculum. Alternatively, if there are many subtasks, one can use a hierarchical or options based framework where individual policies are learned for different subtasks (e.g., kicking, passing, running) and then a higher level agent selects from these low level policies as its action space. Note, this issue may also fall under the “Exploration” section and require explicit exploration mechanisms such as the [Intrinsic Curiosity Module.](https://arxiv.org/pdf/1705.05363.pdf) 
### Are your rewards normalized/standardized?
* Reason: Rewards of magnitudinally larger scale will dominate smaller rewards. Additionally, if per timestep rewards get really large, the targets for value functions will become huge as they are the sum of the per timestep rewards. 
* Prescription: In general, keeping rewards between [-1,1] is good practice. Alternatively, you can use running mean/std instance normalization (e.g., the TorchRL [implementation](https://github.com/pytorch/rl/blob/20b6fc92574959b5edd0a7658e3d45ecadaef2eb/torchrl/envs/transforms/transforms.py#L2313) or the Gym [implementation](https://github.com/openai/gym/blob/master/gym/wrappers/normalize.py)).

## Exploration

### Is value loss going up early in training?
* Reason: Typically, at initialization value estimates are ~0.0. Early in training, an agent will likely be encountering new, unseen extrinsic as it explores and so the value estimates will be wrong and loss goes up.
* Prescription: Increasing exploration via intrinsic rewards or entropy bonuses. Alternatively, making the reward function denser by adding intermediate rewards.
### Are actions (roughly) uniformly/normally random early in training?
* Reason: If no priors are used, a freshly initialized network should be near random. This is important for an agent to achieve proper exploration.
* Prescription: Check the policy network is initialized appropriately and that policy entropy doesn’t drop really quickly.
### Are intrinsic rewards decaying as learning progresses in a [singleton](https://arxiv.org/pdf/2210.05805.pdf) task?
* Reason: Intrinsic rewards are meant to encourage exploration, typically by some measure of novelty. As an agent explores, the value of additional exploration (or revisiting previously explored state-actions) is diminished as novelty decreases.  Ideally, as intrinsic reward starts to go down, extrinsic reward should start to increase. 
* Prescription: Intrinsic rewards should be normalized. If the intrinsic reward has gone to 0 but the agent has not learned anything, one can try slow the dynamics of the intrinsic module (i.e., reduce the learning rate of Random Network Distillation or add noise).
### Are [episodic](https://arxiv.org/pdf/2210.05805.pdf) intrinsic rewards remaining constant or increasing as learning progresses in an episodic task?
* Reason: Intrinsic rewards are meant to encourage exploration, typically by some measure of novelty. In episodic tasks, since novelty may not decrease and exploratory behavior may actually improve, intrinsic rewards should remain constant or increase.
* Prescription:  Extrinsic reward should of course also increase. If that is not the case, it could mean that the two objectives are misaligned and that there is a trade off between the two. If such a trade off is unavoidable, then the extrinsic reward needs to have priority over the episodic bonus. Some ways to achieve this are to use a decaying schedule on the episodic bonus, have separate explore (with episodic bonus only) and exploit (with extrinsic reward only) policies and use the explore policy to generate more diverse starting states for the exploit policy or use behavioral cloning to bootstrap training. Also, intrinsic rewards should be normalized.

## Environment Dynamics

### Can you train a low entropy forward dynamics and/or reward model (also useful for offline RL)?
* Reason: The next state and rewards are used to generate targets for value learning in RL algorithms. If these are very noisy, then the targets will be noisy and learning may be slow or unstable. Environments may be inherently stochastic (i.e., random spawns of enemies), the formulation of the obs space may have a missing variable (i.e., a POMDP) or the dependence on the previous state may just be very loose to nonexistent.
* Prescription: Depending on the source of the noise, it may be useful to revisit the observation formulation to be sure it includes all necessary information, a network architecture that can process the sequence of previous states rather than just the last state (i.e., LSTM, Transformer) or even use a Distributional RL algorithm to explicitly model the distribution of value (rather than just expected value).

## Observation Space

### Are your observations normalized/standardized?
* Reason: Input and output targets that have the same relative scale tend to be more stable as network weights don’t need to get really large/small to compensate.  For the same reason, learning tends to be faster since network weights are initialized to an appropriate scale and don’t need to get there by gradient descent. Additionally, if there is extreme difference in scale between observation features (e.g., [-1,+1] vs. [-1000, 1000]), the larger may dominate the smaller before weights can compensate. 
* Prescription: If you know the minimum/maximum ranges for these values, you can manually normalize to the range of [0,1]. Alternatively, you can use running mean/std instance normalization (e.g., the TorchRL [implementation](https://github.com/pytorch/rl/blob/20b6fc92574959b5edd0a7658e3d45ecadaef2eb/torchrl/envs/transforms/transforms.py#L2313) or the Gym [implementation](https://github.com/openai/gym/blob/master/gym/wrappers/normalize.py)). The mean and std deviation will change radically at the beginning of training but then slowly converge with more data. One can collect a large buffer before making any updates to compute a starting mean and std if stability is a problem.  

## Action Space

### Is the effect of an action changing dramatically during an episode?
* Reason: If an action leads to a failure during the first stages of training, the agent may learn to never perform it and it could prevent it from solving the task entirely (i.e., a ‘submit your work’ action).
* Prescription: It may be that the problem should be formulated hierarchically (i.e. an agent that learns to ‘submit work’). Additionally, sufficient exploration becomes very important in this case.
### Is the action space too high dimensional?
* Reason: If the action space is extremely large (i.e., recommender systems), it may be the case that adequately exploring the entire action space is infeasible.
* Prescription: To alleviate this, one could manually prune the action space or develop state-dependent heuristics to mask/filter which actions are available to the agent (e.g., masking out the “fire” action in certain Atari games or illegal moves in chess) or combine actions/action sequences (e.g., grasp and release actions in manipulation tasks could be the same action and also sequences of primitives). If this is not possible, alternative methods exist such as [top-p](https://arxiv.org/pdf/2210.01241.pdf) sampling wherein you sample from only the top actions with cumulative probability p.
### Are your actions normalized/standardized?
* Reason: Input and output targets that have the same relative scale tend to be more stable as network weights don’t need to get really large/small to compensate. For the same reason, learning tends to be faster since network weights are initialized to an appropriate scale and don’t need to get there by gradient descent. In some algorithms, actions can be input to a Q function and in others gradients can flow directly through the action output into the policy (e.g., reparameterization in Soft Actor-Critic) so it is important for reasonably scaled actions.
* Prescription: It is common to [clip](https://github.com/DLR-RM/stable-baselines3/blob/b702884c23b6aeaa5d2a830b37d6b15fb1bdf983/stable_baselines3/common/policies.py#L354) the action outputs of a policy to a reasonable range. Note, this clipped action should not (as opposed to the raw action) be used for training because the clip operation is not part of the computation graph and gradients will be incorrect. This should be thought of as part of the environment and so a policy will learn that actions in the bounded region lead to higher reward. One can also use a squashing function such as tanh. This can be part of the computation graph and to do this efficiently, one should correct the log probs such as is done [here](https://github.com/Unity-Technologies/ml-agents/blob/develop/ml-agents/mlagents/trainers/torch_entities/distributions.py#L110).  Remember to remap actions to the original action space on the environment side if normalized.
