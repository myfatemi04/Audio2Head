import torch
import os
import ts.protocol.otf_message_handler as otf

"""
Adapted from: https://github.com/pytorch/serve/blob/master/docs/custom_service.md
"""

def load_config_and_opt(base_dir):
    import yaml, argparse

    config_file = r"config/vox-256.yaml"
    parameters_file = r"config/parameters.yaml"

    with open(os.path.join(base_dir, config_file)) as f:
        config = yaml.load(f, yaml.FullLoader)

    with open(os.path.join(base_dir, parameters_file)) as f:
        opt = argparse.Namespace(**yaml.load(f, yaml.FullLoader))

    return (config, opt)

class ModelHandler(object):
    """
    A custom model handler implementation.
    """

    def __init__(self):
        self._context = None
        self.initialized = False
        self.model = None
        self.device = None

    def initialize(self, context):
        """
        Invoke by torchserve for loading a model
        :param context: context contains model server system properties
        :return:
        """

        #  load the model
        self.manifest = context.manifest

        properties = context.system_properties
        model_dir = properties.get("model_dir")
        self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")

        # Read model serialize/pt file
        serialized_file = self.manifest['model']['serializedFile']
        model_pt_path = os.path.join(model_dir, serialized_file)
        if not os.path.isfile(model_pt_path):
            raise RuntimeError("Missing the model.pt file")
        
        import torchserve_model

        config, opt = load_config_and_opt(model_dir)

        self.model = torchserve_model.TorchserveModel(config, opt).to(self.device)
        self.model.load_state_dict(
            torch.load(model_pt_path, map_location=self.device)
        )

        self.initialized = True


    def handle(self, data, context):
        """
        Invoke by TorchServe for prediction request.
        Do pre-processing of data, prediction using model and postprocessing of prediciton output
        :param data: Input data for prediction
        :param context: Initial context contains model server system properties.
        :return: prediction output
        """
        pred_out = self.model.forward(data)

        return self.postprocess(pred_out)
    
    def postprocess(self, inference_output: list):
        # https://github.com/pytorch/serve/blob/master/examples/text_to_speech_synthesizer/waveglow_handler.py
        # input is a list of paths

        results = []
        for filename in inference_output:
            with open(filename, "rb") as f:
                results.append(f.read())

        return results

# # Create model object
# model = None

# def entry_point_function_name(data, context):
#     """
#     Works on data and context to create model object or process inference request.
#     Following sample demonstrates how model object can be initialized for jit mode.
#     Similarly you can do it for eager mode models.
#     :param data: Input data for prediction
#     :param context: context contains model server system properties
#     :return: prediction output
#     """
#     global model

#     if not data:
#         manifest = context.manifest

#         properties = context.system_properties
#         model_dir = properties.get("model_dir")
#         device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")

#         # Read model serialize/pt file
#         serialized_file = manifest['model']['serializedFile']
#         model_pt_path = os.path.join(model_dir, serialized_file)
#         if not os.path.isfile(model_pt_path):
#             raise RuntimeError("Missing the model.pt file")

#         model = torch.jit.load(model_pt_path, map_location=device)
#     else:
#         # infer and return result
#         return model(data)
