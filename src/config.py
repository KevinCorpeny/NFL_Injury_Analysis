from pathlib import Path
from typing import Dict, Any
import yaml
from dataclasses import dataclass

@dataclass
class DataConfig:
    raw_data_path: Path
    processed_data_path: Path
    injury_data_path: Path
    seasons: list[int]

@dataclass
class ModelConfig:
    random_state: int
    test_size: float
    features: list[str]

@dataclass
class Config:
    data: DataConfig
    model: ModelConfig

    @classmethod
    def from_yaml(cls, config_path: str) -> 'Config':
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        data_config = DataConfig(
            raw_data_path=Path(config_dict['data']['raw_data_path']),
            processed_data_path=Path(config_dict['data']['processed_data_path']),
            injury_data_path=Path(config_dict['data']['injury_data_path']),
            seasons=config_dict['data']['seasons']
        )
        
        model_config = ModelConfig(
            random_state=config_dict['model']['random_state'],
            test_size=config_dict['model']['test_size'],
            features=config_dict['model']['features']
        )
        
        return cls(data=data_config, model=model_config) 