from dataclasses import dataclass

@dataclass
class VideosToImagesConfig:
    video_dir: str
    image_dir: str
    output_fps: int
    output_size: tuple[int, int]
