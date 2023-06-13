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
        self.model.eval()

        self.initialized = True


    def handle(self, data, context):
        """
        Invoke by TorchServe for prediction request.
        Do pre-processing of data, prediction using model and postprocessing of prediciton output
        :param data: Input data for prediction
        :param context: Initial context contains model server system properties.
        :return: prediction output
        """
        paths = self.model.forward(data)

        return self.postprocess(paths, context)
    
    def postprocess(self, paths: list, context):
        # https://github.com/pytorch/serve/blob/master/examples/text_to_speech_synthesizer/waveglow_handler.py
        # input is a list of paths

        # import io
        # import cv2

        path = paths[0]

        with open(path, "rb") as f:
            while chunk := f.read(4096):
                otf.send_intermediate_predict_response([chunk], context.request_ids, "intermediate", 200, context)
            
        return [b'']

        # results = []

        # for filename in inference_output:
        #     with open(filename, "rb") as f:
        #         while True:
        #             chunk = f.read(4096)
        #             if len(chunk) == 0:
        #                 break
        #             otf.send_intermediate_predict_response([chunk], context.request_ids, "intermediate", 200, context)
        #         return [b'']

        # return results

# # Create model object
# model = None

# doesn't work
def stream_numpy_images_doesnt_work(images, framerate=25):
    import av

    # Get image dimensions from the first image
    height, width, _ = images[0].shape
    
    # Create a PyAV output container in memory
    container = av.open('output.mp4', 'w')
    
    # Add a video stream to the container
    video_stream = container.add_stream('libx264', rate=framerate)
    video_stream.width = width
    video_stream.height = height
    video_stream.pix_fmt = 'yuv420p'
    
    # Iterate over the numpy images and encode them into video frames
    for image in images:
        # Convert the numpy array to a PyAV video frame
        frame = av.VideoFrame.from_ndarray(image, format='rgb24')
        
        # Encode the frame and add it to the video stream
        packet = video_stream.encode(frame)
        
        # Write the encoded packet to the container
        if packet is not None:
            container.mux(packet)
        
        # Retrieve the byte data of the video from the in-memory container
        output_bytes = container.output[0].to_bytes()
        
        # Yield the video bytes as they become available
        yield output_bytes
    
    # Flush the remaining frames and finalize the container
    container.close()
