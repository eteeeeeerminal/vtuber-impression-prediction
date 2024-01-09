# src/hydra-plugins/random_id_provide.py
# 参考: https://qiita.com/ryuichiastrona/items/003dcf792eabb8a8df95

import uuid
from omegaconf import OmegaConf

class IDGenerator:
    def __init__(self, digit: int = 8) -> None:
        self.digit = digit # 長すぎるので切り捨てる桁数を指定。
        self.uid = str(uuid.uuid4())[: self.digit]

    def generate(self) -> str:
        return self.uid
id_generator = IDGenerator()

OmegaConf.register_new_resolver("experiment_id", id_generator.generate)
