from dataclasses import dataclass

@dataclass
class ModelLogConfig:
    run_n: int
    logname: str
    logdir: str
    legend: str
    output: str
    legend_title: str
    y_min: float
    y_max: float
