import os
import pathlib

import tqdm
import hydra
import cv2

from preprocess import VideosToImagesConfig

def video_to_imgs(
    video_file: pathlib.Path, output_dir: pathlib.Path,
    output_fps: int, output_size: tuple[int, int]
    ):

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    video_file = cv2.VideoCapture(str(video_file))
    assert video_file.isOpened() == True

    original_fps: int = video_file.get(cv2.CAP_PROP_FPS)
    period = int(original_fps / output_fps)
    if period <= 0:
        # 元動画のfpsが6fpsというすごく低い動画があったため
        period = 1

    i = 0
    output_i = 0
    ret, frame = video_file.read()
    while frame is not None:
        if i % period == 0:
            resized_image = cv2.resize(frame, output_size)
            cv2.imwrite(str(output_dir.joinpath(f"{output_i}.jpg")), resized_image)
            output_i += 1

        ret, frame = video_file.read()
        i += 1


@hydra.main(version_base=None, config_path="../configs", config_name="videos_to_images")
def main(cfg: VideosToImagesConfig):
    video_dir = pathlib.Path(cfg.video_dir)
    image_dir = pathlib.Path(cfg.image_dir)
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    video_files = list(video_dir.glob("*.mp4"))
    for video_file in tqdm.tqdm(
        video_files,
        total=len(video_files), ncols=80, leave=False
    ):
        video_name = video_file.name
        print(f" Converting: {video_name}")
        output_dir = image_dir.joinpath(video_name.replace(".mp4", ""))
        video_to_imgs(video_file, output_dir, cfg.output_fps, cfg.output_size)

if __name__ == "__main__":
    main()
