# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from tensordict import TensorDict
import math
from torchrl.modules.mcts.scores import UCBScore, PUCTScore, EXP3Score

# Sample TensorDict for testing
def create_node(num_actions, weights=None, batch_size=None, device="cpu", custom_keys=None):
    if custom_keys is None:
        custom_keys = {
            "num_actions_key": "num_actions",
            "weights_key": "weights",
            "score_key": "score",
        }

    if batch_size:
        data = {custom_keys["num_actions_key"]: torch.tensor([num_actions] * batch_size, device=device)}
        if weights is not None:
            if weights.ndim == 1:
                weights = weights.unsqueeze(0).repeat(batch_size, 1)
            data[custom_keys["weights_key"]] = weights.to(device)
        td = TensorDict(data, batch_size=[batch_size], device=device)
    else:
        data = {custom_keys["num_actions_key"]: torch.tensor(num_actions, device=device)}
        if weights is not None:
            data[custom_keys["weights_key"]] = weights.to(device)
        td = TensorDict(data, batch_size=[], device=device)
    return td

# Sample TensorDict node for UCBScore
def create_ucb_node(win_count, visits, total_visits, batch_size=None, device="cpu", custom_keys=None):
    if custom_keys is None:
        custom_keys = {
            "win_count_key": "win_count",
            "visits_key": "visits",
            "total_visits_key": "total_visits",
            "score_key": "score",
        }
    
    win_count = torch.as_tensor(win_count, device=device, dtype=torch.float32)
    visits = torch.as_tensor(visits, device=device, dtype=torch.float32)
    total_visits = torch.as_tensor(total_visits, device=device, dtype=torch.float32)

    if batch_size:
        if win_count.ndim == 0:
            win_count = win_count.unsqueeze(0).repeat(batch_size)
        elif win_count.shape[0] != batch_size:
             raise ValueError("Batch size mismatch for win_count")
        if visits.ndim == 0:
            visits = visits.unsqueeze(0).repeat(batch_size)
        elif visits.shape[0] != batch_size:
             raise ValueError("Batch size mismatch for visits")
        if total_visits.ndim == 0:
            total_visits = total_visits.unsqueeze(0).repeat(batch_size)
        elif total_visits.shape[0] != batch_size and total_visits.numel() != 1 :
             raise ValueError("Batch size mismatch for total_visits")
        if total_visits.numel() == 1 and batch_size > 1:
            total_visits = total_visits.repeat(batch_size)
    
        data = {
            custom_keys["win_count_key"]: win_count,
            custom_keys["visits_key"]: visits,
            custom_keys["total_visits_key"]: total_visits,
        }
        td = TensorDict(data, batch_size=[batch_s for batch_s in batch_size] if isinstance(batch_size, (list, tuple)) else [batch_size], device=device)
    else:
        data = {
            custom_keys["win_count_key"]: win_count,
            custom_keys["visits_key"]: visits,
            custom_keys["total_visits_key"]: total_visits,
        }
        td = TensorDict(data, batch_size=win_count.shape[:-1] if win_count.ndim > 1 else [], device=device)
    
    return td

# Helper function to create a sample TensorDict node for PUCTScore
def create_puct_node(win_count, visits, total_visits, prior_prob, batch_size=None, device="cpu", custom_keys=None):
    if custom_keys is None:
        custom_keys = {
            "win_count_key": "win_count",
            "visits_key": "visits",
            "total_visits_key": "total_visits",
            "prior_prob_key": "prior_prob",
            "score_key": "score",
        }

    win_count = torch.as_tensor(win_count, device=device, dtype=torch.float32)
    visits = torch.as_tensor(visits, device=device, dtype=torch.float32)
    total_visits = torch.as_tensor(total_visits, device=device, dtype=torch.float32)
    prior_prob = torch.as_tensor(prior_prob, device=device, dtype=torch.float32)

    if batch_size:
        if win_count.ndim == 0: win_count = win_count.unsqueeze(0).repeat(batch_size)
        elif win_count.shape[0] != batch_size: raise ValueError("Batch size mismatch for win_count")
        if visits.ndim == 0: visits = visits.unsqueeze(0).repeat(batch_size)
        elif visits.shape[0] != batch_size: raise ValueError("Batch size mismatch for visits")
        if prior_prob.ndim == 0: prior_prob = prior_prob.unsqueeze(0).repeat(batch_size)
        elif prior_prob.shape[0] != batch_size: raise ValueError("Batch size mismatch for prior_prob")
        
        if total_visits.numel() == 1 and batch_size > 1: # scalar total_visits for batch
            total_visits = total_visits.repeat(batch_size)
        elif total_visits.ndim == 0 : total_visits = total_visits.unsqueeze(0).repeat(batch_size) # make it (batch_size,)
        elif total_visits.shape[0] != batch_size : raise ValueError("Batch size mismatch for total_visits")


        data = {
            custom_keys["win_count_key"]: win_count,
            custom_keys["visits_key"]: visits,
            custom_keys["total_visits_key"]: total_visits,
            custom_keys["prior_prob_key"]: prior_prob,
        }
        if isinstance(batch_size, (list, tuple)):
            td_batch_size = batch_size
        else:
            td_batch_size = [batch_size]
        td = TensorDict(data, batch_size=td_batch_size, device=device)

    else:
        data = {
            custom_keys["win_count_key"]: win_count,
            custom_keys["visits_key"]: visits,
            custom_keys["total_visits_key"]: total_visits,
            custom_keys["prior_prob_key"]: prior_prob,
        }
        td_batch_size = win_count.shape[:-1] if win_count.ndim > 1 else []

        td = TensorDict(data, batch_size=td_batch_size, device=device)

    return td

class TestEXP3Score:
    @pytest.fixture
    def default_scorer(self):
        return EXP3Score()
    
    @pytest.fixture
    def custom_key_names(self):
        return {
            "weights_key": "custom_weights",
            "score_key": "custom_scores",
            "num_actions_key": "custom_num_actions",
            "action_prob_key": "custom_actions_prob",
            "reward_key": "custom_reward"
        }
    
    @pytest.mark.parametrize("gamma_val", [0.1, 0.5, 0.9])
    def test_initialization(self, gamma_val):
        scorer = EXP3Score(gamma=gamma_val)
        assert scorer.gamma == gamma_val
        scorer_default = EXP3Score()
        assert scorer_default.gamma == 0.1

    def test_forward_initial_weights(self, default_scorer):
        num_actions = 3
        node = create_node(num_actions=num_actions)
        
        default_scorer.forward(node)

        assert default_scorer.weights_key in node.keys()
        expected_weights = torch.ones(num_actions)
        torch.testing.assert_close(node.get(default_scorer.weights_key), expected_weights)

        expected_scores = torch.ones(num_actions) / num_actions
        torch.testing.assert_close(node.get(default_scorer.score_key), expected_scores)
        torch.testing.assert_close(node.get(default_scorer.score_key).sum(), torch.tensor(1.0))

    def test_forward_custom_weights(self, default_scorer):
        num_actions = 3
        weights = torch.tensor([1.0, 2.0, 3.0])
        node = create_node(num_actions=num_actions, weights=weights)

        default_scorer.forward(node)
        
        gamma = default_scorer.gamma
        sum_w = weights.sum()
        expected_scores = (1 - gamma) * (weights / sum_w) + (gamma / num_actions)
        
        torch.testing.assert_close(node.get(default_scorer.score_key), expected_scores)
        torch.testing.assert_close(node.get(default_scorer.score_key).sum(), torch.tensor(1.0))

    @pytest.mark.parametrize("batch_s", [2, 4])
    def test_forward_batch(self, default_scorer, batch_s):
        num_actions = 3
        node_initial = create_node(num_actions=num_actions, batch_size=batch_s)
        default_scorer.forward(node_initial)

        expected_weights_initial = torch.ones(batch_s, num_actions)
        torch.testing.assert_close(node_initial.get(default_scorer.weights_key), expected_weights_initial)
        
        expected_scores_initial = torch.ones(batch_s, num_actions) / num_actions
        torch.testing.assert_close(node_initial.get(default_scorer.score_key), expected_scores_initial)
        torch.testing.assert_close(node_initial.get(default_scorer.score_key).sum(dim=-1), torch.ones(batch_s))

        weights_custom = torch.rand(batch_s, num_actions) + 0.1
        node_custom = create_node(num_actions=num_actions, weights=weights_custom, batch_size=batch_s)
        default_scorer.forward(node_custom)

        gamma = default_scorer.gamma
        sum_w_custom = weights_custom.sum(dim=-1, keepdim=True)
        expected_scores_custom = (1 - gamma) * (weights_custom / sum_w_custom) + (gamma / num_actions)
        torch.testing.assert_close(node_custom.get(default_scorer.score_key), expected_scores_custom, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(node_custom.get(default_scorer.score_key).sum(dim=-1), torch.ones(batch_s))
        
    def test_update_weights_single_node(self, default_scorer):
        num_actions = 3
        action_idx = 0
        reward = 1.0
        node = create_node(num_actions=num_actions)

        default_scorer.forward(node)
        initial_weights = node.get(default_scorer.weights_key).clone()
        prob_i = node.get(default_scorer.score_key)[action_idx]

        default_scorer.update_weights(node, action_idx, reward)
        
        updated_weights = node.get(default_scorer.weights_key)
        gamma = default_scorer.gamma
        k = num_actions
        
        expected_new_weight_val = initial_weights[action_idx] * math.exp((gamma / k) * (reward / prob_i))
        
        torch.testing.assert_close(updated_weights[action_idx], torch.tensor(expected_new_weight_val))
        torch.testing.assert_close(updated_weights[action_idx+1:], initial_weights[action_idx+1:])

        default_scorer.forward(node)
        sum_w_updated = updated_weights.sum()
        expected_scores_after_update = (1-gamma)*(updated_weights/sum_w_updated) + (gamma/k)
        torch.testing.assert_close(node.get(default_scorer.score_key), expected_scores_after_update)


    def test_update_weights_zero_reward(self, default_scorer):
        num_actions = 3
        action_idx = 1
        reward = 0.0
        weights = torch.tensor([1.0, 2.0, 1.5])
        node = create_node(num_actions=num_actions, weights=weights)

        default_scorer.forward(node)
        initial_weights = node.get(default_scorer.weights_key).clone()
        prob_i = node.get(default_scorer.score_key)[action_idx]

        default_scorer.update_weights(node, action_idx, reward)
        updated_weights = node.get(default_scorer.weights_key)
        gamma = default_scorer.gamma
        k = num_actions

        expected_new_weight_val = initial_weights[action_idx] * math.exp((gamma / k) * (reward / prob_i)) 
        torch.testing.assert_close(updated_weights[action_idx], expected_new_weight_val)
        torch.testing.assert_close(updated_weights[action_idx], initial_weights[action_idx])

    @pytest.mark.parametrize("batch_s", [2, 3])
    def test_update_weights_batch(self, default_scorer, batch_s):
        num_actions = 3
        node = create_node(num_actions=num_actions, batch_size=batch_s)
        default_scorer.forward(node)

        initial_weights_batch = node.get(default_scorer.weights_key).clone()
        probs_batch = node.get(default_scorer.score_key).clone()
        
        rewards = torch.rand(batch_s)
        action_indices = torch.randint(0, num_actions, (batch_s,))

        expected_updated_weights_batch = initial_weights_batch.clone()
        gamma = default_scorer.gamma
        k = num_actions

        for i in range(batch_s):
            action_idx = action_indices[i].item()
            reward = rewards[i].item()
            
            single_node_td = node[i]
            
            current_weight_item = initial_weights_batch[i, action_idx]
            prob_i_item = probs_batch[i, action_idx]
            
            exp_val = math.exp((gamma / k) * (reward / prob_i_item))
            expected_updated_weights_batch[i, action_idx] = current_weight_item * exp_val

            node_item_to_update = node[i:i+1]
            default_scorer.update_weights(node_item_to_update, action_idx, reward)

        torch.testing.assert_close(node.get(default_scorer.weights_key), expected_updated_weights_batch, atol=1e-5, rtol=1e-5)

    def test_single_action(self, default_scorer):
        num_actions = 1
        node = create_node(num_actions=num_actions)
        default_scorer.forward(node)

        assert default_scorer.weights_key in node.keys()
        torch.testing.assert_close(node.get(default_scorer.weights_key), torch.ones(num_actions))
        torch.testing.assert_close(node.get(default_scorer.score_key), torch.ones(num_actions)) # p_i = 1.0
        
        action_idx = 0
        reward = 0.5
        initial_weights = node.get(default_scorer.weights_key).clone()
        prob_i = node.get(default_scorer.score_key)[action_idx]

        default_scorer.update_weights(node, action_idx, reward)
        updated_weights = node.get(default_scorer.weights_key)
        gamma = default_scorer.gamma
        k = num_actions
        
        expected_new_weight_val = initial_weights[action_idx] * math.exp((gamma / k) * (reward / prob_i))
        torch.testing.assert_close(updated_weights[action_idx], torch.tensor(expected_new_weight_val))

    @pytest.mark.parametrize("gamma_val, expected_behavior", [
        (0.0, "exploitation"), (1.0, "exploration")
    ])
    def test_gamma_extremes(self, gamma_val, expected_behavior):
        scorer = EXP3Score(gamma=gamma_val)
        num_actions = 3
        weights = torch.tensor([1.0, 2.0, 7.0])
        node = create_node(num_actions=num_actions, weights=weights)
        
        scorer.forward(node)
        scores = node.get(scorer.score_key)

        if expected_behavior == "exploitation":
            expected_scores = weights / weights.sum()
            torch.testing.assert_close(scores, expected_scores)
        elif expected_behavior == "exploration":
            expected_scores = torch.ones(num_actions) / num_actions
            torch.testing.assert_close(scores, expected_scores)

    def test_custom_keys(self, custom_key_names):
        gamma = 0.2
        scorer = EXP3Score(
            gamma=gamma,
            weights_key=custom_key_names["weights_key"],
            score_key=custom_key_names["score_key"],
            num_actions_key=custom_key_names["num_actions_key"],
            action_prob_key=custom_key_names["action_prob_key"],
        )
        num_actions = 2

        node1 = create_node(num_actions=num_actions, custom_keys=custom_key_names)
        scorer.forward(node1)
        
        assert custom_key_names["weights_key"] in node1.keys()
        expected_weights1 = torch.ones(num_actions)
        torch.testing.assert_close(node1.get(custom_key_names["weights_key"]), expected_weights1)
        expected_scores1 = torch.ones(num_actions) / num_actions
        torch.testing.assert_close(node1.get(custom_key_names["score_key"]), expected_scores1)
        if scorer.action_prob_key != scorer.score_key: # Check if action_prob_key was also populated
             torch.testing.assert_close(node1.get(custom_key_names["action_prob_key"]), expected_scores1)

        weights2_val = torch.tensor([1.0, 3.0])
        node2 = create_node(num_actions=num_actions, weights=weights2_val, custom_keys=custom_key_names)
        scorer.forward(node2)
        
        sum_w2 = weights2_val.sum()
        expected_scores2 = (1 - gamma) * (weights2_val / sum_w2) + (gamma / num_actions)
        torch.testing.assert_close(node2.get(custom_key_names["score_key"]), expected_scores2)

        action_idx = 0
        reward = 1.0
        initial_weights2 = node2.get(custom_key_names["weights_key"]).clone()
        prob_i2 = node2.get(custom_key_names["score_key"])[action_idx]
        
        scorer.update_weights(node2, action_idx, reward)
        updated_weights2 = node2.get(custom_key_names["weights_key"])
        k = num_actions
        
        expected_new_weight_val2 = initial_weights2[action_idx] * math.exp((gamma / k) * (reward / prob_i2))
        torch.testing.assert_close(updated_weights2[action_idx], torch.tensor(expected_new_weight_val2))

    def test_forward_raises_error_on_mismatched_num_actions(self, default_scorer):
        num_actions_prop = 3
        weights = torch.tensor([1.0, 2.0, 3.0, 4.0]) # K=4 from weights
        node = create_node(num_actions=num_actions_prop, weights=weights) # num_actions=3

        with pytest.raises(ValueError, match="Shape of weights .* implies 4 actions, but num_actions is 3"):
            default_scorer.forward(node)

        weights_ok = torch.tensor([1.0, 2.0, 3.0])
        node_ok = create_node(num_actions=torch.tensor(4), weights=weights_ok) # num_actions=4 from tensor

        with pytest.raises(ValueError, match="Shape of weights .* implies 3 actions, but num_actions is 4"):
            default_scorer.forward(node_ok)

    def test_update_weights_handles_prob_zero(self, default_scorer):
        num_actions = 2
        action_idx = 0
        reward = 1.0
        scorer_exploit = EXP3Score(gamma=0.0)
        weights = torch.tensor([0.0, 1.0])
        node = create_node(num_actions=num_actions, weights=weights)

        scorer_exploit.forward(node) # p_0 will be 0
        assert node.get(scorer_exploit.score_key)[0] == 0.0

        with pytest.warns(UserWarning, match="Probability p_i\\(t\\) for action 0 is 0.0"):
            scorer_exploit.update_weights(node, action_idx, reward)
        torch.testing.assert_close(node.get(scorer_exploit.weights_key)[action_idx], torch.tensor(0.0))

    def test_init_raises_error_gamma_out_of_range(self):
        with pytest.raises(ValueError, match="gamma must be between 0 and 1"):
            EXP3Score(gamma=-0.1)
        with pytest.raises(ValueError, match="gamma must be between 0 and 1"):
            EXP3Score(gamma=1.1)

    def test_update_weights_reward_warning(self, default_scorer):
        num_actions = 2
        node = create_node(num_actions=num_actions)
        default_scorer.forward(node)
        with pytest.warns(UserWarning, match="Reward .* is outside the expected \\[0,1\\] range"):
            default_scorer.update_weights(node, 0, 1.5)
        with pytest.warns(UserWarning, match="Reward .* is outside the expected \\[0,1\\] range"):
            default_scorer.update_weights(node, 0, -0.5)
        initial_weight = node.get(default_scorer.weights_key)[0].clone()
        default_scorer.update_weights(node, 0, 1.5)
        assert node.get(default_scorer.weights_key)[0] != initial_weight # it changed


class TestUCBScore:
    @pytest.fixture
    def default_ucb_scorer(self):
        return UCBScore(c=math.sqrt(2))

    @pytest.fixture
    def ucb_custom_key_names(self):
        return {
            "win_count_key": "custom_wins",
            "visits_key": "custom_visits",
            "total_visits_key": "custom_total_visits",
            "score_key": "custom_ucb_score",
        }

    @pytest.mark.parametrize("c_val", [0.5, 1.0, math.sqrt(2), 5.0])
    def test_initialization(self, c_val):
        scorer = UCBScore(c=c_val)
        assert scorer.c == c_val

    def test_forward_basic(self, default_ucb_scorer):
        win_count = torch.tensor([10.0, 5.0, 20.0])
        visits = torch.tensor([15.0, 10.0, 25.0])
        total_visits_parent = torch.tensor(50.0)

        node = create_ucb_node(win_count=win_count, visits=visits, total_visits=total_visits_parent)
        default_ucb_scorer.forward(node)

        c = default_ucb_scorer.c
        exploitation_term = win_count / visits
        exploration_term = c * total_visits_parent.sqrt() / (1 + visits)
        expected_scores = exploitation_term + exploration_term
        
        torch.testing.assert_close(node.get(default_ucb_scorer.score_key), expected_scores)

    def test_forward_zero_visits(self, default_ucb_scorer):
        win_count = torch.tensor([0.0, 0.0])
        visits = torch.tensor([10.0, 0.0])
        total_visits_parent = torch.tensor(10.0)

        node = create_ucb_node(win_count=win_count, visits=visits, total_visits=total_visits_parent)
        default_ucb_scorer.forward(node)

        c = default_ucb_scorer.c
        scores = node.get(default_ucb_scorer.score_key)

        expected_score_0 = (win_count[0] / visits[0]) + c * total_visits_parent.sqrt() / (1 + visits[0])
        torch.testing.assert_close(scores[0], expected_score_0)
        assert torch.isnan(scores[1]), "Score for unvisited action (0 visits, 0 wins) should be NaN due to 0/0, unless handled."

    @pytest.mark.parametrize("batch_s", [2, 3])
    def test_forward_batch(self, default_ucb_scorer, batch_s):
        win_count = torch.rand(batch_s, 2) * 10
        visits = torch.rand(batch_s, 2) * 5 + 1 
        total_visits_parent = torch.rand(batch_s) * 20 + float(batch_s)

        node = create_ucb_node(win_count=win_count, visits=visits, total_visits=total_visits_parent, batch_size=batch_s)
        default_ucb_scorer.forward(node)

        c = default_ucb_scorer.c
        exploitation_term = win_count / visits
        exploration_term = c * total_visits_parent.unsqueeze(-1).sqrt() / (1 + visits)
        expected_scores = exploitation_term + exploration_term
        
        torch.testing.assert_close(node.get(default_ucb_scorer.score_key), expected_scores)

    def test_forward_exploration_term(self, default_ucb_scorer):
        win_count = torch.tensor([0.0, 0.0, 0.0])
        visits = torch.tensor([10.0, 5.0, 1.0])
        total_visits_parent = torch.tensor(100.0)

        node = create_ucb_node(win_count=win_count, visits=visits, total_visits=total_visits_parent)
        default_ucb_scorer.forward(node)

        c = default_ucb_scorer.c
        expected_scores = c * total_visits_parent.sqrt() / (1 + visits)
        
        torch.testing.assert_close(node.get(default_ucb_scorer.score_key), expected_scores)

    def test_custom_keys(self, ucb_custom_key_names):
        c_val = 1.5
        scorer = UCBScore(
            c=c_val,
            win_count_key=ucb_custom_key_names["win_count_key"],
            visits_key=ucb_custom_key_names["visits_key"],
            total_visits_key=ucb_custom_key_names["total_visits_key"],
            score_key=ucb_custom_key_names["score_key"],
        )
        
        win_count = torch.tensor([1.0, 2.0])
        visits = torch.tensor([3.0, 4.0])
        total_visits_parent = torch.tensor(10.0)

        node = create_ucb_node(
            win_count=win_count, 
            visits=visits, 
            total_visits=total_visits_parent,
            custom_keys=ucb_custom_key_names
        )
        scorer.forward(node)

        exploitation = win_count / visits
        exploration = c_val * total_visits_parent.sqrt() / (1 + visits)
        expected_scores = exploitation + exploration
        
        assert ucb_custom_key_names["score_key"] in node.keys()
        torch.testing.assert_close(node.get(ucb_custom_key_names["score_key"]), expected_scores)
        
        assert "score" not in node.keys()
        assert "win_count" not in node.keys() 
        assert "visits" not in node.keys()
        assert "total_visits" not in node.keys()


class TestPUCTScore:
    @pytest.fixture
    def default_puct_scorer(self):
        return PUCTScore(c=5.0)

    @pytest.fixture
    def puct_custom_key_names(self):
        return {
            "win_count_key": "custom_puct_wins",
            "visits_key": "custom_puct_visits",
            "total_visits_key": "custom_puct_total_visits",
            "prior_prob_key": "custom_puct_priors",
            "score_key": "custom_puct_score",
        }

    @pytest.mark.parametrize("c_val", [0.5, 1.0, 5.0, 10.0])
    def test_initialization(self, c_val):
        scorer = PUCTScore(c=c_val)
        assert scorer.c == c_val

    def test_forward_basic(self, default_puct_scorer):
        win_count = torch.tensor([10.0, 5.0, 20.0])
        visits = torch.tensor([15.0, 10.0, 25.0])  
        prior_prob = torch.tensor([0.4, 0.3, 0.3]) 
        total_visits_parent = torch.tensor(50.0)

        node = create_puct_node(
            win_count=win_count, 
            visits=visits, 
            total_visits=total_visits_parent, 
            prior_prob=prior_prob
        )
        default_puct_scorer.forward(node)

        c = default_puct_scorer.c
        exploitation_term = win_count / visits
        exploration_term = c * prior_prob * total_visits_parent.sqrt() / (1 + visits)
        expected_scores = exploitation_term + exploration_term
        
        torch.testing.assert_close(node.get(default_puct_scorer.score_key), expected_scores)

    def test_forward_zero_visits(self, default_puct_scorer):
        win_count = torch.tensor([0.0, 0.0])     
        visits = torch.tensor([10.0, 0.0])        
        prior_prob = torch.tensor([0.6, 0.4])
        total_visits_parent = torch.tensor(10.0)

        node = create_puct_node(
            win_count=win_count, 
            visits=visits, 
            total_visits=total_visits_parent,
            prior_prob=prior_prob
        )
        default_puct_scorer.forward(node)

        c = default_puct_scorer.c
        scores = node.get(default_puct_scorer.score_key)

        expected_score_0 = (win_count[0] / visits[0]) + \
                           c * prior_prob[0] * total_visits_parent.sqrt() / (1 + visits[0])
        torch.testing.assert_close(scores[0], expected_score_0)

        assert torch.isnan(scores[1]), "Score for unvisited action (0 visits, 0 wins) should be NaN due to 0/0, unless handled."


    @pytest.mark.parametrize("batch_s", [2, 3])
    def test_forward_batch(self, default_puct_scorer, batch_s):
        num_actions = 2
        win_count = torch.rand(batch_s, num_actions) * 10
        visits = torch.rand(batch_s, num_actions) * 5 + 1
        prior_prob = torch.rand(batch_s, num_actions)
        prior_prob = prior_prob / prior_prob.sum(dim=-1, keepdim=True)
        total_visits_parent = torch.rand(batch_s) * 20 + float(batch_s)

        node = create_puct_node(
            win_count=win_count, 
            visits=visits, 
            total_visits=total_visits_parent, 
            prior_prob=prior_prob,
            batch_size=batch_s
        )
        default_puct_scorer.forward(node)

        c = default_puct_scorer.c
        exploitation_term = win_count / visits
        exploration_term = c * prior_prob * total_visits_parent.unsqueeze(-1).sqrt() / (1 + visits)
        expected_scores = exploitation_term + exploration_term
        
        torch.testing.assert_close(node.get(default_puct_scorer.score_key), expected_scores, atol=1e-6, rtol=1e-6)

    def test_forward_exploration_term(self, default_puct_scorer):
        num_actions = 3
        win_count = torch.zeros(num_actions)
        visits = torch.tensor([10.0, 5.0, 1.0])
        prior_prob = torch.tensor([0.3, 0.5, 0.2])
        total_visits_parent = torch.tensor(100.0)

        node = create_puct_node(
            win_count=win_count, 
            visits=visits, 
            total_visits=total_visits_parent,
            prior_prob=prior_prob
        )
        default_puct_scorer.forward(node)

        c = default_puct_scorer.c
        # exploitation_term is effectively 0
        expected_scores = c * prior_prob * total_visits_parent.sqrt() / (1 + visits)
        
        torch.testing.assert_close(node.get(default_puct_scorer.score_key), expected_scores)

    def test_custom_keys(self, puct_custom_key_names):
        c_val = 2.5
        scorer = PUCTScore(
            c=c_val,
            win_count_key=puct_custom_key_names["win_count_key"],
            visits_key=puct_custom_key_names["visits_key"],
            total_visits_key=puct_custom_key_names["total_visits_key"],
            prior_prob_key=puct_custom_key_names["prior_prob_key"],
            score_key=puct_custom_key_names["score_key"],
        )
        
        win_count = torch.tensor([1.0, 2.0])
        visits = torch.tensor([3.0, 4.0])
        prior_prob = torch.tensor([0.5, 0.5])
        total_visits_parent = torch.tensor(10.0)

        node = create_puct_node(
            win_count=win_count, 
            visits=visits, 
            total_visits=total_visits_parent,
            prior_prob=prior_prob,
            custom_keys=puct_custom_key_names
        )
        scorer.forward(node)

        exploitation = win_count / visits
        exploration = c_val * prior_prob * total_visits_parent.sqrt() / (1 + visits)
        expected_scores = exploitation + exploration
        
        assert puct_custom_key_names["score_key"] in node.keys()
        torch.testing.assert_close(node.get(puct_custom_key_names["score_key"]), expected_scores)
        
        # Check that default keys are not present
        assert "score" not in node.keys()
        assert "win_count" not in node.keys()
        assert "visits" not in node.keys()
        assert "total_visits" not in node.keys()
        assert "prior_prob" not in node.keys()