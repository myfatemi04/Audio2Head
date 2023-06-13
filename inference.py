import argparse
import subprocess
import python_speech_features
from scipy.io import wavfile
from scipy.interpolate import interp1d
import numpy as np
import pyworld
import torch
from modules.audio2pose import get_pose_from_audio
from skimage import io, img_as_float32
import cv2
from modules.generator import OcclusionAwareGenerator
from modules.keypoint_detector import KPDetector
from modules.audio2kp import AudioModel3D
import yaml,os,imageio
import tqdm
# from preloaded_model_manager import kp_detector, generator, audio2kp

def draw_annotation_box( image, rotation_vector, translation_vector, color=(255, 255, 255), line_width=2):
    """Draw a 3D box as annotation of pose"""

    camera_matrix = np.array(
        [[233.333, 0, 128],
         [0, 233.333, 128],
         [0, 0, 1]], dtype="double")

    dist_coeefs = np.zeros((4, 1))

    point_3d = []
    rear_size = 75
    rear_depth = 0
    point_3d.append((-rear_size, -rear_size, rear_depth))
    point_3d.append((-rear_size, rear_size, rear_depth))
    point_3d.append((rear_size, rear_size, rear_depth))
    point_3d.append((rear_size, -rear_size, rear_depth))
    point_3d.append((-rear_size, -rear_size, rear_depth))

    front_size = 100
    front_depth = 100
    point_3d.append((-front_size, -front_size, front_depth))
    point_3d.append((-front_size, front_size, front_depth))
    point_3d.append((front_size, front_size, front_depth))
    point_3d.append((front_size, -front_size, front_depth))
    point_3d.append((-front_size, -front_size, front_depth))
    point_3d = np.array(point_3d, dtype=np.float32).reshape(-1, 3)

    # Map to 2d image points
    (point_2d, _) = cv2.projectPoints(point_3d,
                                      rotation_vector,
                                      translation_vector,
                                      camera_matrix,
                                      dist_coeefs)
    point_2d = np.int32(point_2d.reshape(-1, 2))

    # Draw all the lines
    cv2.polylines(image, [point_2d], True, color, line_width, cv2.LINE_AA)
    cv2.line(image, tuple(point_2d[1]), tuple(
        point_2d[6]), color, line_width, cv2.LINE_AA)
    cv2.line(image, tuple(point_2d[2]), tuple(
        point_2d[7]), color, line_width, cv2.LINE_AA)
    cv2.line(image, tuple(point_2d[3]), tuple(
        point_2d[8]), color, line_width, cv2.LINE_AA)

def inter_pitch(y,y_flag):
    frame_num = y.shape[0]
    i = 0
    last = -1
    while(i<frame_num):
        if y_flag[i] == 0:
            while True:
                if y_flag[i]==0:
                    if i == frame_num-1:
                        if last !=-1:
                            y[last+1:] = y[last]
                        i+=1
                        break
                    i+=1
                else:
                    break
            if i >= frame_num:
                break
            elif last == -1:
                y[:i] = y[i]
            else:
                inter_num = i-last+1
                fy = np.array([y[last],y[i]])
                fx = np.linspace(0, 1, num=2)
                f = interp1d(fx,fy)
                fx_new = np.linspace(0,1,inter_num)
                fy_new = f(fx_new)
                y[last+1:i] = fy_new[1:-1]
                last = i
                i+=1

        else:
            last = i
            i+=1
    return y

def get_audio_feature_from_audio(audio_path,norm = True):
    sample_rate, audio = wavfile.read(audio_path)
    if len(audio.shape) == 2:
        if np.min(audio[:, 0]) <= 0:
            audio = audio[:, 1]
        else:
            audio = audio[:, 0]
    if norm:
        audio = audio - np.mean(audio)
        audio = audio / np.max(np.abs(audio))
        a = python_speech_features.mfcc(audio, sample_rate)
        b = python_speech_features.logfbank(audio, sample_rate)
        c, _ = pyworld.harvest(audio, sample_rate, frame_period=10)
        c_flag = (c == 0.0) ^ 1
        c = inter_pitch(c, c_flag)
        c = np.expand_dims(c, axis=1)
        c_flag = np.expand_dims(c_flag, axis=1)
        frame_num = np.min([a.shape[0], b.shape[0], c.shape[0]])

        cat = np.concatenate([a[:frame_num], b[:frame_num], c[:frame_num], c_flag[:frame_num]], axis=1)
        return cat

@torch.no_grad()
def audio2head(request_id, img_path, kp_detector, generator, audio2kp, audio2pose):
    import time

    mp3_audio_path = f"./requests/{request_id}.mp3"
    wav_audio_path = f"./requests/{request_id}.wav"
    raw_video_path = f"./requests/{request_id}.raw.mp4"
    final_video_path = f"./requests/{request_id}.final.mp4"

    t_start = time.time()

    # Convert mp3 to wav
    command = ("ffmpeg -y -i %s -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 %s" % (mp3_audio_path, wav_audio_path))
    subprocess.call(command, shell=True, stdout=None)

    t_video_conversion = time.time() - t_start

    audio_feature = get_audio_feature_from_audio(wav_audio_path)
    frames = len(audio_feature) // 4

    t_audio_feature = time.time() - t_start

    img = io.imread(img_path)[:, :, :3]
    img = cv2.resize(img, (256, 256))

    img = np.array(img_as_float32(img))
    img = img.transpose((2, 0, 1))
    # img: (1, 3, 256, 256)
    img = torch.from_numpy(img).unsqueeze(0).cuda()

    ref_pose_rot, ref_pose_trans = get_pose_from_audio(img, audio_feature, audio2pose)
    torch.cuda.empty_cache()

    t_pose = time.time() - t_start

    opt = argparse.Namespace(**yaml.load(open("./config/parameters.yaml"), yaml.FullLoader))
    audio_f = []
    poses = []
    pad = np.zeros((4,41),dtype=np.float32)
    for i in range(0, frames, opt.seq_len // 2):
        wav_audio_path = []
        temp_pos = []
        for j in range(opt.seq_len):
            if i + j < frames:
                wav_audio_path.append(audio_feature[(i+j)*4:(i+j)*4+4])
                trans = ref_pose_trans[i + j]
                rot = ref_pose_rot[i + j]
            else:
                wav_audio_path.append(pad)
                trans = ref_pose_trans[-1]
                rot = ref_pose_rot[-1]

            pose = np.zeros([256, 256])
            draw_annotation_box(pose, np.array(rot), np.array(trans))
            temp_pos.append(pose)
        audio_f.append(wav_audio_path)
        poses.append(temp_pos)

    t_annotations = time.time() - t_start
        
    audio_f = torch.from_numpy(np.array(audio_f,dtype=np.float32)).unsqueeze(0)
    poses = torch.from_numpy(np.array(poses, dtype=np.float32)).unsqueeze(0)

    num_batches = audio_f.shape[1]
    predictions_gen = []
    processed_frames = 0

    generated_keypoints = {
        "value": [],
        "jacobian": [],
    }
    
    # {value: (b, c, 2), jacobian_map: (b, j, 4, h, w), jacobian: (b, j, 2, 2), pred_fature: (b, f, h, w)}
    # where b = 1 for the initial image
    kp_gen_source = kp_detector(img)
    for batch_index in tqdm.tqdm(range(num_batches), desc='Generating keypoints...'):
        # same format as before, but for some reason, the first dimension is 1 for all of them, and the second dimension
        # is the batch size.
        gen_kp = audio2kp({
            "audio": audio_f[:, batch_index].cuda(),
            "pose": poses[:, batch_index].cuda(),
            "id_img": img,
        })

        # Unsure why this is necessary, but it is.
        # Oh wait actually, it's probably due to padded audio.
        # We select the first 3/4, then then section from 1/4 to 3/4, etc., skipping ahead by 1/2 the window size every time.
        if batch_index == 0:
            startid = 0
            end_id = opt.seq_len // 4 * 3
        else:
            startid = opt.seq_len // 4
            end_id = opt.seq_len // 4 * 3

        generated_keypoints['value'].append(gen_kp['value'][0][startid:end_id])
        generated_keypoints['jacobian'].append(gen_kp['jacobian'][0][startid:end_id])

    # concatenate all generated keypoints and run them through the generator in batches
    generated_keypoints['value'] = torch.cat(generated_keypoints['value'], dim=0)
    generated_keypoints['jacobian'] = torch.cat(generated_keypoints['jacobian'], dim=0)

    t_keypoints = time.time() - t_start

    render_batch_size = 64
    start_id = 0
    total_frames_ = generated_keypoints['value'].shape[0]
    with tqdm.tqdm("Rendering...", total=total_frames_ // render_batch_size) as pbar:
        while start_id < total_frames_:
            end_id = min(total_frames_, start_id + render_batch_size)

            ones = [1, 1, 1, 1, 1, 1]

            num_frames = end_id - start_id
            img_batch = img.repeat(num_frames, *ones[:len(img.shape) - 1])
            # this code is horrid. im so sorry.
            kp_gen_source_batch = {
                'value': kp_gen_source['value'].repeat(num_frames, *ones[:len(kp_gen_source['value'].shape) - 1]),
                'jacobian': kp_gen_source['jacobian'].repeat(num_frames, *ones[:len(kp_gen_source['jacobian'].shape) - 1]),
                'jacobian_map': kp_gen_source['jacobian_map'].repeat(num_frames, *ones[:len(kp_gen_source['jacobian_map'].shape) - 1]),
                'pred_fature': kp_gen_source['pred_fature'].repeat(num_frames, *ones[:len(kp_gen_source['pred_fature'].shape) - 1]),
            }
            tt = {
                'value': generated_keypoints['value'][start_id:end_id],
                'jacobian': generated_keypoints['jacobian'][start_id:end_id],
            }
            print('tt_value.shape:', tt['value'].shape)
            print('tt_jacobian.shape:', tt['jacobian'].shape)

            out_gen = generator(img_batch, kp_source=kp_gen_source_batch, kp_driving=tt)
            out_gen["kp_source"] = kp_gen_source_batch
            out_gen["kp_driving"] = tt
            del out_gen['sparse_deformed']
            del out_gen['occlusion_map']
            del out_gen['deformed']
            
            # # YIELD a prediction
            # yield (np.transpose(out_gen['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0] * 255).astype(np.uint8)
            predictions_gen.extend(
                (np.transpose(out_gen['prediction'].data.cpu().numpy(), [0, 2, 3, 1]) * 255).astype(np.uint8)
            )

            start_id += render_batch_size
            pbar.update()

    predictions_gen = predictions_gen[:frames]
    
    t_render = time.time() - t_start
    
    # save video
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    video = cv2.VideoWriter(raw_video_path, fourcc, 25, predictions_gen[0].shape[:2][::-1])
    for image in predictions_gen:
        # switch rgb color channels
        video.write(image[:, :, ::-1])
    video.release()

    cmd = r'ffmpeg -y -i "%s" -i "%s" -vcodec libx264 "%s"' % (raw_video_path, mp3_audio_path, final_video_path)
    os.system(cmd)

    t_total = time.time() - t_start
    t_per_frame = t_total / total_frames_

    print(f"{t_video_conversion=:.2f} {t_audio_feature=:.2f} {t_pose=:.2f} {t_annotations=:.2f} {t_keypoints=:.2f} {t_render=:.2f} {t_total=:.2f} {t_per_frame=:.2f}")
    
    return final_video_path
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_path",default=r"./demo/audio/intro.wav",help="audio file sampled as 16k hz")
    parser.add_argument("--img_path",default=r"./demo/img/paint.jpg", help="reference image")
    parser.add_argument("--save_path",default=r"./results", help="save path")
    parser.add_argument("--model_path",default=r"./checkpoints/audio2head.pth.tar", help="pretrained model path")

    parse = parser.parse_args()

    os.makedirs(parse.save_path,exist_ok=True)
    audio2head(parse.audio_path,parse.img_path,parse.model_path,parse.save_path)