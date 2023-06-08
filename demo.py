import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GenerationConfig

from tensordict import TensorDict

tokenizer = GPT2Tokenizer.from_pretrained('gpt2', padding_side='left')
tokenizer.pad_token=tokenizer.eos_token

texts = ["Replace me by any text you'd like.", "aha, what is this? It's a"]
encoded_input = tokenizer(texts, return_tensors='pt', padding=True)
model = GPT2LMHeadModel.from_pretrained('gpt2')
generation_config = GenerationConfig(output_scores=True,
                                     return_dict_in_generate=True,
                                     padding=True,
                                     max_length=100,
                                     repetition_penalty=0.5,
                                     do_sample=True,
                                    )

generation_config.pad_token_id = tokenizer.eos_token_id
generation_config.eos_token_id = tokenizer.eos_token_id

encoded_input = tokenizer(texts, return_tensors='pt', return_token_type_ids=True, padding=True)
out = model.generate(**encoded_input, generation_config=generation_config)

def generated_to_tensordict(obj, *, tokenizer=None, done_token=None):
    inputs = {}
    if done_token is None:
        done_token = tokenizer.eos_token_id

    # scores
    num_gens = len(obj.scores)

    inputs["sample_log_prob"] = torch.stack(list(obj.scores), 1)
    seq = obj.sequences.unfold(1, num_gens, 1).transpose(-2, -1)
    inputs["sequences"] = seq[..., :-1]
    inputs["mask"] = inputs["sequences"] != done_token
    inputs["next", "sequences"] = seq[..., 1:]
    inputs["next", "mask"] = inputs["next", "sequences"] != done_token
    inputs["action"] = seq[..., -1]

    # filter out the dones that follow a done
    inputs["next", "done"] = inputs["action"] == done_token
    inputs["next", "done"][..., 1:][inputs["next", "done"][..., :-1].clone()] = False
    inputs["next", "done"] = inputs["next", "done"].unsqueeze(-1)
    inputs["next", "truncated"] = torch.zeros_like(inputs["next", "done"])
    inputs["next", "truncated"][..., -1, :] = ~(inputs["next", "done"].any(-2))
    return TensorDict(inputs, batch_size=inputs["action"].shape[:2])


td = generated_to_tensordict(out, tokenizer=tokenizer)
print(td)
