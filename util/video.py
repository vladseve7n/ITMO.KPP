import os
import glob
from os import path as osp
import mimetypes
import subprocess
import numpy as np
import cv2
import ffmpeg
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt


def save_image(image_path, img):
    plt.imsave(image_path, img)


def load_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def convert_images_to_video(images_path, videoname, fps=30, audio_path=None, img_ext='jpg'):
    """Convert folder with images to the video.

    Parameters
    ----------
    images_path : str
        Path to images with target images. 
        Image names should be in format *[0-9]*.png.
    videoname : str
        Name of the output videofile (.mp4 video)
    fps : int, optional
        Frames per second (FPS), by default 30
    audio_path : str, optional
        Path to the audiofile to unite with video, by default None
    """
    # get all images
    images_names = os.listdir(images_path)
    if '.ipynb_checkpoints' in images_names:
        images_names.remove('.ipynb_checkpoints')
    images_names = sorted(glob.glob(os.path.join(images_path, f'*[0-9]*.{img_ext}')),key=lambda x: int(x.split('/')[-1].split('.')[0]))
    
    # get image size, using information from sample image
    image_size = load_image(images_names[0]).shape
    image_size = image_size[:2][::-1]

    # video.mp4 or video.avi -> video preprocessing
    # videoname_without_ext, ext = os.path.splitext(os.path.basename(videoname))
    output_dir = os.path.dirname(videoname)
    # videoname_avi = os.path.join(output_dir, videoname_without_ext + '.avi')

    # define video object and writer
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    videoWriter = cv2.VideoWriter(videoname, fourcc, fps, image_size)
    # write image
    for index in tqdm(range(len(images_names))):
        frame = cv2.imread(images_names[index])
        videoWriter.write(frame)
    videoWriter.release()

    # overlay audio
    if audio_path is not None:
        
        video_without_ext, ext = os.path.splitext(videoname)
        temporary_video = video_without_ext + "_sound" + ext
        os.system(
            f'ffmpeg -y -i {videoname} -i {audio_path} -c:v copy -c:a aac -map 0:v -map 1:a {temporary_video}'
        )
        os.system(
            f'mv {temporary_video} {videoname}'
        )

def get_video_meta_info(video_path):
    ret = {}
    probe = ffmpeg.probe(video_path)
    video_streams = [stream for stream in probe['streams'] if stream['codec_type'] == 'video']
    has_audio = any(stream['codec_type'] == 'audio' for stream in probe['streams'])
    ret['width'] = video_streams[0]['width']
    ret['height'] = video_streams[0]['height']
    ret['fps'] = eval(video_streams[0]['avg_frame_rate'])
    ret['audio'] = ffmpeg.input(video_path).audio if has_audio else None
    ret['nb_frames'] = int(video_streams[0]['nb_frames'])
    return ret


def get_sub_video(args, num_process, process_idx):
    if num_process == 1:
        return args.input
    meta = get_video_meta_info(args.input)
    duration = int(meta['nb_frames'] / meta['fps'])
    part_time = duration // num_process
    print(f'duration: {duration}, part_time: {part_time}')
    os.makedirs(osp.join(args.output, f'{args.video_name}_inp_tmp_videos'), exist_ok=True)
    out_path = osp.join(args.output, f'{args.video_name}_inp_tmp_videos', f'{process_idx:03d}.mp4')
    cmd = [
        args.ffmpeg_bin, f'-i {args.input}', '-ss', f'{part_time * process_idx}',
        f'-to {part_time * (process_idx + 1)}' if process_idx != num_process - 1 else '', '-async 1', out_path, '-y'
    ]
    print(' '.join(cmd))
    subprocess.call(' '.join(cmd), shell=True)
    return out_path