# Audio2Head: Audio-driven One-shot Talking-head Generation with Natural Head Motion (IJCAI 2021)

## Serving This Model

Written by: Michael Fatemi

First, you must `pip install torchserve torch-model-archiver`.

Then, create a model archive using `create_model_archive.py`. Note that this must be done when modifying the model weights, the model inference code, and the custom handler code.

This model requires a custom handler, which was created according to [this guide](https://github.com/pytorch/serve/blob/master/docs/custom_service.md).

This also requires Java, which you can install with `sudo apt-get install -y openjdk-17-jdk`.

Configuration for TorchServe is in `config.properties`, according to [this guide](https://pytorch.org/serve/configuration.html).

In addition to default dependencies, metrics are collected with `nvgpu` (`pip install nvgpu`).

#### [Paper](https://www.ijcai.org/proceedings/2021/0152.pdf) | [Demo](https://www.youtube.com/watch?v=xvcBJ29l8rA)

#### Requirements

- Python 3.6 , Pytorch >= 1.6 and ffmpeg

- Other requirements are listed in the 'requirements.txt'

  

#### Pretrained Checkpoint

Please download the pretrained checkpoint from [google-drive](https://drive.google.com/file/d/1tvI43ZIrnx9Ti2TpFiEO4dK5DOwcECD7/view?usp=sharing) and put it within the folder (`/checkpoints`).



#### Generate Demo Results

```
python inference.py --audio_path xxx.wav --img_path xxx.jpg
```

Note that the input images must keep the same height and width and the face should be appropriately cropped as in `/demo/img`.



#### License and Citation

```
@InProceedings{wang2021audio2head,
author = Suzhen Wang, Lincheng Li, Yu Ding, Changjie Fan, Xin Yu
title = {Audio2Head: Audio-driven One-shot Talking-head Generation with Natural Head Motion},
booktitle = {the 30th International Joint Conference on Artificial Intelligence (IJCAI-21)},
year = {2021},
}
```



#### Acknowledgement

This codebase is based on [First Order Motion Model](https://github.com/AliaksandrSiarohin/first-order-model), thanks for their contribution.





