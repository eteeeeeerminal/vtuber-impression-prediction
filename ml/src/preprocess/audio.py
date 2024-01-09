from dataclasses import dataclass

@dataclass
class LogFBankConfig:
    sample_rate: int
    winlen: float
    winstep: float
    nfilt: int
    nfft: int
    lowfreq: int
    preemph: float

@dataclass
class MFCCConfig:
    sample_rate: int
    winlen: float
    winstep: float
    numcep: int
    nfilt: int
    nfft: int
    lowfreq: int
    preemph: int

@dataclass
class VideosToAudioFeaturesConfig:
    video_dir: str
    audio_features_dir: str
    logfbank: LogFBankConfig
    mfcc: MFCCConfig