from typing import Any, Mapping
import torch
import yaml
import argparse
import random
import os
import elevenlabs

config_file = r"./config/vox-256.yaml"
parameters_file = r"./config/parameters.yaml"

with open(config_file) as f:
    config = yaml.load(f, yaml.FullLoader)

with open(parameters_file) as f:
    opt = argparse.Namespace(**yaml.load(f, yaml.FullLoader))

class TorchserveModel(torch.nn.Module):
    def __init__(self):
        from modules.generator import OcclusionAwareGenerator
        from modules.keypoint_detector import KPDetector
        from modules.audio2kp import AudioModel3D
        from modules.audio2pose import audio2poseLSTM

        super().__init__()

        self.kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                            **config['model_params']['common_params'])
        self.generator = OcclusionAwareGenerator(**config['model_params']['generator_params'],
                                            **config['model_params']['common_params'])
        self.audio2kp = AudioModel3D(opt)
        self.audio2pose = audio2poseLSTM()

    def load_state_dict(self, checkpoint: Mapping[str, Any], strict: bool = True):
        self.kp_detector.load_state_dict(checkpoint["kp_detector"], strict)
        self.generator.load_state_dict(checkpoint["generator"], strict)
        self.audio2kp.load_state_dict(checkpoint["audio2kp"], strict)
        self.audio2pose.load_state_dict(checkpoint["audio2pose"], strict)

    def forward(self, text: str):
        from inference import audio2head

        # generate driving audio
        audio = elevenlabs.generate(text, api_key=os.environ['ELEVENLABS_API_KEY'], voice='AUfmNX7a1AJ7HNn4TKft')
        
        request_id = "request_" + str(random.randint(0, 1000000000))

        if not os.path.exists("requests"):
            os.mkdir("requests")

        # save to mp3
        with open(f'./requests/{request_id}.mp3', 'wb') as f:
            f.write(audio)

        path = audio2head(request_id, 'pavel.png', self.kp_detector, self.generator, self.audio2kp, self.audio2pose)

        print(path)

# Prevent Torchserve from being confused about which `torch.nn.Module` it should be serving.
# Import statements above add ambiguity.
__all__ = ['TorchserveModel']
