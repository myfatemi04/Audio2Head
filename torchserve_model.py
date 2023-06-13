from typing import Any, Mapping
import torch
import random
import os
import elevenlabs

class TorchserveModel(torch.nn.Module):
    # config and opt are passed in from `synthesis_entrypoint.py`
    def __init__(self, config, opt):
        from modules.generator import OcclusionAwareGenerator
        from modules.keypoint_detector import KPDetector
        from modules.audio2kp import AudioModel3D
        from modules.audio2pose import audio2poseLSTM
        # from modules.generator import OcclusionAwareGenerator
        # from modules.keypoint_detector import KPDetector
        # from modules.audio2kp import AudioModel3D
        # from modules.audio2pose import audio2poseLSTM

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

    def forward(self, instances: list):
        from inference import audio2head

        assert len(instances) == 1, "Only one text input is supported at this time."

        with open("11.key", "r") as f:
            api_key = f.read().strip()

        text = instances[0]['body']

        # generate driving audio
        audio = elevenlabs.generate(text, api_key=api_key, voice='AUfmNX7a1AJ7HNn4TKft')
        
        request_id = "request_" + str(random.randint(0, 1000000000))

        if not os.path.exists("requests"):
            os.mkdir("requests")

        # save to mp3
        with open(f'./requests/{request_id}.mp3', 'wb') as f:
            f.write(audio)

        path = audio2head(request_id, 'pavel.png', self.kp_detector, self.generator, self.audio2kp, self.audio2pose)

        return [path]

# Prevent Torchserve from being confused about which `torch.nn.Module` it should be serving.
# Import statements above add ambiguity.
__all__ = ['TorchserveModel']
