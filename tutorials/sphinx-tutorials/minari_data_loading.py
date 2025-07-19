import torch
from torchrl._utils import logger as torchrl_logger
from torchrl.data.datasets.minari_data import MinariExperienceReplay

# Define the possible missions
COLORS = ["red", "green", "blue", "purple", "yellow", "grey"]
OBJECT_TYPES = ["box", "ball", "key"]
MISSION_TO_IDX = {
    f"pick up {color} {obj}": i
    for i, (color, obj) in enumerate((c, o) for c in COLORS for o in OBJECT_TYPES)
}
NUM_MISSIONS = len(MISSION_TO_IDX)


# Define the encoder function
def encode_mission_string(mission: bytes) -> torch.Tensor:
    mission = mission.decode("utf-8")
    clean_mission = mission.replace(" a", "").replace(" the", "")
    idx = MISSION_TO_IDX.get(clean_mission, -1)
    if idx == -1:
        raise ValueError(f"Unknown mission string: {clean_mission}")
    return torch.nn.functional.one_hot(torch.tensor(idx), num_classes=NUM_MISSIONS).to(
        torch.uint8
    )


def main():
    # Download the dataset and apply the string-to-tensor map
    data = MinariExperienceReplay(
        dataset_id="minigrid/BabyAI-Pickup/optimal-v0",
        batch_size=1,
        download="force",
        string_to_tensor_map={"observations/mission": encode_mission_string},
    )

    # Sample and inspect
    print("Sampling data with transformed mission field:")
    sample = data.sample(batch_size=1)
    torchrl_logger.info(sample)


if __name__ == "__main__":
    main()
