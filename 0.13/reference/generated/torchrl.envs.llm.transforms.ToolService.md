# ToolService

*class*torchrl.envs.llm.transforms.ToolService(**args*, ***kwargs*)[[source]](../../_modules/torchrl/envs/llm/transforms/tools.html#ToolService)

Protocol for side-effecting service callable with structured IO.

A tool service is a callable that can be invoked with keyword arguments
and returns a dictionary of results. It has a name and input/output schemas.

Variables:

- **name** (*str*) - The name of the tool service.
- **schema_in** (*dict**[**str**,**Any**]*) - Input schema describing expected parameters.
- **schema_out** (*dict**[**str**,**Any**]*) - Output schema describing returned data.