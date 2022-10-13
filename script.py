s = """        eps_init: float = 1.0,
        eps_end: float = 0.1,
        annealing_num_steps: int = 1000,
        action_key: str = "action",
        spec: Optional[TensorSpec] = None,"""

s = s.split("\n")
for i, c in enumerate(s):
    c = c.strip()
    l, r = c.find(":"), c.find("=") + 2

    s[i] = "- " + c[:l] + ": " + c[r:-1]
print("\n".join(s))
