from gym.envs.classic_control.pendulum import PendulumEnv

class PendulumWithSafety(PendulumEnv):
    def step(self, u):
        out = super().step(u)
        sin, cos, vel = out[0]
        safe = sin > 0 and cos > 0 # some quadrant is considered safe
        safety_cost = abs(sin) * float(sin <= 0) + abs(cos) * float(cos <= 0)
        out[-1]["safety_cost"] = safety_cost
        out[-1]["safe"] = safe
        return out
