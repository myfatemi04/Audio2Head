import argparse
import yaml
from modules.generator import OcclusionAwareGenerator
from modules.keypoint_detector import KPDetector
from modules.audio2kp import AudioModel3D
from modules.audio2pose import audio2poseLSTM
import torch

# Preload the Torch linalg module.
torch.inverse(torch.ones((0, 0), device="cuda:0"))

model_path = r"./checkpoints/audio2head.pth.tar"

config_file = r"./config/vox-256.yaml"

with open(config_file) as f:
    config = yaml.load(f, yaml.FullLoader)

kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                            **config['model_params']['common_params'])
generator = OcclusionAwareGenerator(**config['model_params']['generator_params'],
                                    **config['model_params']['common_params'])
kp_detector = kp_detector.cuda()
generator = generator.cuda()
audio2pose = audio2poseLSTM().cuda()

opt = argparse.Namespace(**yaml.load(open("./config/parameters.yaml"), yaml.FullLoader))
audio2kp = AudioModel3D(opt).cuda()

checkpoint  = torch.load(model_path)
kp_detector.load_state_dict(checkpoint["kp_detector"])
generator.load_state_dict(checkpoint["generator"])
audio2kp.load_state_dict(checkpoint["audio2kp"])
audio2pose.load_state_dict(checkpoint["audio2pose"])

generator.eval()
kp_detector.eval()
audio2kp.eval()
audio2pose.eval()