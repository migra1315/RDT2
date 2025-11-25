# Run under root directory of our repo
import yaml
import torch
import numpy as np
from models.rdt_inferencer import RDTInferencer


with open("configs/rdt/post_train.yaml", "r") as f:
  model_config = yaml.safe_load(f)

model = RDTInferencer(
  config=model_config,
  pretrained_path="robotics-diffusion-transformer/RDT2-FM",
  # TODO: modify `normalizer_path` to your own downloaded normalizer path
  # download from http://ml.cs.tsinghua.edu.cn/~lingxuan/rdt2/umi_normalizer_wo_downsample_indentity_rot.pt
  normalizer_path="robotics-diffusion-transformer/umi_normalizer_wo_downsample_indentity_rot.pt",  
  pretrained_vision_language_model_name_or_path="robotics-diffusion-transformer/RDT2-VQ", # use RDT2-VQ as the VLM backbone
  device="cuda:0",
  dtype=torch.bfloat16,
)

# 384×384 RGB, uint8
H, W = 384, 384
camera0_np = np.random.randint(0, 256, size=(H, W, 3), dtype=np.uint8)
camera1_np = np.random.randint(0, 256, size=(H, W, 3), dtype=np.uint8)

result = model.step(
    observations={
        'images': {
            'left_stereo': camera0_np, # left arm RGB image in np.ndarray of shape (384, 384, 3) with dtype=np.uint8
            'right_stereo': camera1_np, # right arm RGB image in np.ndarray of shape (384, 384, 3) with dtype=np.uint8
        },
        # use zero input current state for currently
        # preserve input interface for future fine-tuning
        'state': np.zeros(model_config["common"]["state_dim"]).astype(np.float32)
    },
    instruction="Pick up the apple." # Language instruction
    # We suggest using Instruction in format "verb + object" with Capitalized First Letter and trailing period 
)


# relative action chunk in np.ndarray of shape (24, 20) with dtype=np.float32
# with the same format as RDT2-VQ
action_chunk = result.detach().cpu().numpy()

# rescale gripper width from [0, 0.088] to [0, 0.1]
for robot_idx in range(2):
    action_chunk[:, robot_idx * 10 + 9] = action_chunk[:, robot_idx * 10 + 9] / 0.088 * 0.1
    print(action_chunk)