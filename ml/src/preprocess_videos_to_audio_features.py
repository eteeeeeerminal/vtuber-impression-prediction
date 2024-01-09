# opencv で mp4 から音声を array 読み込み
# python hogehoge で特徴量抽出
# 特徴量のハイパラ考える
# とりあえずlogfbank でやってみる

import os
import pathlib

import tqdm
import hydra
import numpy as np
from pydub import AudioSegment
from python_speech_features import logfbank, mfcc

from preprocess.audio import VideosToAudioFeaturesConfig

def audio_to_monaural_array(audio):
    data = np.array(audio.get_array_of_samples())
    if audio.channels == 1:
        return data
    elif audio.channels == 2:
        return np.mean(np.array([data[::2], data[1::2]]), axis=0)

def video_to_audio(cfg: VideosToAudioFeaturesConfig, mode: str):
    assert mode in ["logfbank", "mfcc"]
    video_dir = pathlib.Path(cfg.video_dir)
    feature_dir = pathlib.Path(cfg.audio_features_dir).joinpath(mode)
    os.makedirs(feature_dir, exist_ok=True)

    video_files = list(video_dir.glob("*.mp4"))
    for video_file in tqdm.tqdm(
        video_files,
        total=len(video_files), ncols=80, leave=False
    ):
        video_name = video_file.name
        print(video_name)
        audio = AudioSegment.from_file(video_file, "mp4")
        audio_array = audio_to_monaural_array(audio)

        if mode == "logfbank":
            assert cfg.logfbank.sample_rate == audio.frame_rate
            feature = logfbank(audio_array,
                cfg.logfbank.sample_rate,
                cfg.logfbank.winlen, cfg.logfbank.winstep,
                cfg.logfbank.nfilt, cfg.logfbank.nfft,
                cfg.logfbank.lowfreq, preemph = cfg.logfbank.preemph
            )
        elif mode == "mfcc":
            assert cfg.logfbank.sample_rate == audio.frame_rate
            feature = mfcc(audio_array,
                cfg.mfcc.sample_rate,
                cfg.mfcc.winlen, cfg.mfcc.winstep,
                cfg.mfcc.numcep, cfg.mfcc.nfilt, cfg.mfcc.nfft,
                cfg.mfcc.lowfreq, preemph=cfg.mfcc.preemph
            )

        np.savetxt(feature_dir.joinpath(video_name.replace(".mp4", "")+".txt"), feature)


@hydra.main(version_base=None, config_path="../configs", config_name="videos_to_audio_features")
def main(cfg: VideosToAudioFeaturesConfig):
    video_to_audio(cfg, "logfbank")
    video_to_audio(cfg, "mfcc")

if __name__ == "__main__":
    main()

