from __future__ import annotations

from pathlib import Path
from typing import Optional, Any
import yaml
from dataclasses import dataclass, field

@dataclass
class DataConfig:
    raw_data_path: Path
    processed_data_path: Path
    injury_data_path: Path
    seasons: list[int]
    play_data_path: Optional[Path] = None

@dataclass
class FeatureConfig:
    play_features: list[str] = field(default_factory=list)
    injury_features: list[str] = field(default_factory=list)

@dataclass
class ProcessingConfig:
    fill_missing_values: bool = True
    add_derived_features: bool = True
    remove_duplicates: bool = True

@dataclass
class ModelConfig:
    random_state: int
    test_size: float
    features: list[str] = field(default_factory=list)

@dataclass
class ValidationConfig:
    required_columns: list[str] = field(default_factory=list)
    allowed_columns: list[str] = field(default_factory=list)
    column_types: list[str] = field(default_factory=list)
    value_ranges: dict[str, dict[str, Any]] = field(
        default_factory=dict
    )
    consistency_rules: list[dict[str, Any]] = field(default_factory=list)
    cross_field_rules: list[dict[str, Any]] = field(default_factory=list)
    statistical_rules: dict[str, dict[str, Any]] = field(default_factory=dict)

@dataclass
class Config:
    data: DataConfig
    features: FeatureConfig
    processing: ProcessingConfig
    validation: ValidationConfig
    model: ModelConfig
    project_root: Path

    @classmethod
    def from_yaml(cls, config_path: str | Path = "config/default.yaml") -> 'Config':
        config_file = Path(config_path).expanduser().resolve()
        if not config_file.is_file():
            raise FileNotFoundError('Configuration file not found: {config_file}')

        with config_file.open('r', encoding='utf-8') as f:
            raw = yaml.safe_load(f) or dict()

        if not isinstance(raw, dict):
            raise TypeError(f'{config_file} file does not contain a dictionary')

        root = (
            config_file.parent.parent
            if config_file.parent.name == "config"
            else config_file.parent
        )

        def resolve_path(value: str) -> Path:
            path = Path(value).expanduser()
            if path.is_absolute():
                return path
            else:
                return root / path

        data_raw = raw.get('data', {})
        features_raw = raw.get('features', {})
        model_raw = raw.get('model', {})
        processing_raw = raw.get('processing', {})
        validation_raw = raw.get('validation', {})

        data = DataConfig(
            raw_data_path=resolve_path(
                data_raw['raw_data_path']
            ),
            processed_data_path=resolve_path(
                data_raw['processed_data_path']
            ),
            injury_data_path=resolve_path(
                data_raw['injury_data_path']
            ),
            seasons = [
                int(season) for season in data_raw.get('seasons', [])
            ],
            play_data_path = (
                resolve_path(data_raw['play_data_path'])
                if data_raw.get('play_data_path')
                else None
            ),
        )

        features = FeatureConfig(
            play_features=data_raw.get('play_features', []),
            injury_features=data_raw.get('injury_features', []),
        )

        model = ModelConfig(
            random_state = int(data_raw.get('random_state', 42)),
            test_size = int(data_raw.get('test_size', 0.2)),
            features = model_raw.get('features', []),
        )

        processing = ProcessingConfig(
            fill_missing_values = processing_raw.get('fill_missing_values', True),
            add_derived_features = processing_raw.get('add_derived_features', True),
            remove_duplicates = processing_raw.get('remove_duplicates', True),
        )

        validation = ValidationConfig(
            required_columns = validation_raw.get('required_columns', []),
            allowed_columns = validation_raw.get('allowed_columns', []),
            column_types = validation_raw.get('column_types', []),
            value_ranges = validation_raw.get('value_ranges', {}),
            consistency_rules = validation_raw.get('consistency_rules', []),
            cross_field_rules= validation_raw.get('cross_field_rules', []),
            statistical_rules = validation_raw.get('statistical_rules', []),
        )

        if not data.seasons:
            raise ValueError("'data.seasons' must not be empty")

        if not 0 < model.test_size < 1:
            raise ValueError("'model.test_size' must be between 0 and 1")

        return cls(
            data=data,
            features=features,
            processing=processing,
            validation=validation,
            model=model,
            project_root=Path(root.parent),
        )
