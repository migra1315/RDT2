import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
import numpy as np
from vqvae.models.multivqvae import MultiVQVAE
from models.normalizer import LinearNormalizer
from utils import batch_predict_action

# assuming using gpu 0
device = "cuda:0"


processor = AutoProcessor.from_pretrained("robotics-diffusion-transformer/Qwen2.5-VL-7B-Instruct")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "robotics-diffusion-transformer/RDT2-VQ",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map=device
).eval()
vae = MultiVQVAE.from_pretrained("robotics-diffusion-transformer/RVQActionTokenizer").eval()
vae = vae.to(device=device, dtype=torch.float32)

valid_action_id_length = (
    vae.pos_id_len + vae.rot_id_len + vae.grip_id_len
)
# TODO: modify to your own downloaded normalizer path
# download from http://ml.cs.tsinghua.edu.cn/~lingxuan/rdt2/umi_normalizer_wo_downsample_indentity_rot.pt
normalizer = LinearNormalizer.load("robotics-diffusion-transformer/umi_normalizer_wo_downsample_indentity_rot.pt")  # 

# 384×384 RGB, uint8
H, W = 384, 384
camera0_np = np.random.randint(0, 256, size=(1, H, W, 3), dtype=np.uint8)
camera1_np = np.random.randint(0, 256, size=(1, H, W, 3), dtype=np.uint8)

# 转成 torch.uint8 张量，并塞进 list（preprocess_data_from_umi 需要 list）
camera0_torch = torch.from_numpy(camera0_np)      # shape: [H, W, 3]
camera1_torch = torch.from_numpy(camera1_np)      # shape: [H, W, 3]

result = batch_predict_action(
    model,
    processor,
    vae,
    normalizer,
    examples=[
        {
            "obs": {
                # NOTE: following the setting of UMI, camera0_rgb for right arm, camera1_rgb for left arm
                "camera0_rgb": camera0_torch, # right arm RGB image in np.ndarray of shape (1, 384, 384, 3) with dtype=np.uint8
                "camera1_rgb": camera1_torch, # left arm RGB image in np.ndarray of shape (1, 384, 384, 3) with dtype=np.uint8
            },
            "meta": {
                "num_camera": 2
            }
        },
        #...,    # we support batch inference, so you can pass a list of examples
    ],
    valid_action_id_length=valid_action_id_length,
    apply_jpeg_compression=True,
    # Since model is trained with mostly jpeg images, we suggest toggle this on for better formance
    instruction="Pick up the apple."
    # We suggest using Instruction in format "verb + object" with Capitalized First Letter and trailing period 
)

print (result)

# get the predict action from example 0
action_chunk = result["action_pred"][0] # torch.FloatTensor of shape (24, 20) with dtype=torch.float32
# action_chunk (T, D) with T=24, D=20
#   T=24: our action_chunk predicts the future 0.8s in fps=30, i.e. 24 frames
#   D=20: following the setting of UMI, we predict the action for both arms from right to left
#   - [0-2]: RIGHT ARM end effector position in x, y, z (unit: m)
#   - [3-8]: RIGHT ARM end effector rotation in 6D rotation representation
#   - [9]: RIGHT ARM gripper width (unit: m)
#   - [10-12]: LEFT ARM end effector position in x, y, z (unit: m)
#   - [13-18]: LEFT ARM end effector rotation in 6D rotation representation
#   - [19]: LEFT ARM gripper width (unit: m)
# rescale gripper width from [0, 0.088] to [0, 0.1]
for robot_idx in range(2):
    action_chunk[:, robot_idx * 10 + 9] = action_chunk[:, robot_idx * 10 + 9] / 0.088 * 0.1
    