from pathlib import Path

from datasets import load_dataset
from models.transformer import init_transformer
from transformers import AutoTokenizer, GenerationConfig, RepetitionPenaltyLogitsProcessor
from utils import load_and_update_config

HERE = Path(__file__).parent


def main():
    config = load_and_update_config("config/train_rlhf.yaml")
    config["compile"] = False
    # config["device"] = "cpu"
    # config["block_size"] = 512
    # ######## INIT MODELS ########
    model = init_transformer(config)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    generation_config = GenerationConfig(
        max_length=1024,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )

    dataset = load_dataset("CarperAI/openai_summarize_tldr", split="test")
    example = dataset[2]
    tokenized_prompt = tokenizer(example["prompt"], return_tensors="pt").to("cuda")

    res = model.module.generate(
        **tokenized_prompt,
        generation_config=generation_config,
        logits_processor=[RepetitionPenaltyLogitsProcessor(penalty=1.2)],
    )

    string_to_write = (
        "Query: \n"
        f"{example['prompt']}\n"
        f"Generated: \n"
        f"{tokenizer.decode(res[0, len(tokenized_prompt['input_ids'][0]):].cpu().numpy().tolist())}\n"
        f"True label: \n"
        f"{example['label']}\n"
        f"====================================================\n"
    )
    print(string_to_write)


if __name__ == "__main__":
    main()
