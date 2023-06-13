import os
import shutil

"""
Uses torch-model-archiver, according to the following tutorial:

https://github.com/pytorch/serve/blob/master/docs/getting_started.md#install-torchserve-and-torch-model-archiver

"""

# modules_folder = ','.join(['modules/' + fname for fname in os.listdir("modules") if fname != '__pycache__'])

if os.path.exists("tmp"):
    # remove recursively
    os.system("rm -rf tmp")

os.mkdir("tmp")
shutil.copytree("config", "tmp/config", ignore=shutil.ignore_patterns('__pycache__'))
shutil.copytree("modules", "tmp/modules", ignore=shutil.ignore_patterns('__pycache__'))
shutil.copytree("sync_batchnorm", "tmp/sync_batchnorm", ignore=shutil.ignore_patterns('__pycache__'))
shutil.copyfile("inference.py", "tmp/inference.py")

os.system(
    "torch-model-archiver "
    "--model-name audio2head "
    "--version 1.0 "
    "--model-file torchserve_model.py "
    "--serialized-file checkpoints/audio2head.pth.tar "
    "--handler synthesis_entrypoint "
    f"--extra-files \"tmp/,pavel.png,11.key\" "
    "-f"
)

os.system("rm -rf tmp")

# torch-model-archiver --model-name densenet161 --version 1.0 --model-file examples/image_classifier/densenet_161/model.py --serialized-file densenet161-8d451a50.pth --extra-files examples/image_classifier/index_to_name.json --handler image_classifier
