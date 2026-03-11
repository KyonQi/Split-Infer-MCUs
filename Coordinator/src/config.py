"""Runtime configuration for the Coordinator system."""

from dataclasses import dataclass, field


@dataclass
class CoordinatorConfig:
    """Central runtime configuration — constructed from CLI args and passed through the system."""

    host: str = '192.168.1.10'
    port: int = 54321
    num_workers: int = 4
    execution_mode: str = 'block'               # 'block' | 'layer' | 'hybrid'
    model_config_path: str = './src/model_config.json'
    log_level: str = 'INFO'
