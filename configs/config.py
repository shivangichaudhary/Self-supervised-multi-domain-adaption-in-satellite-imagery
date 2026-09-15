import os
import copy
import yaml

DEFAULT_CONFIG = {
    "experiment_name": "sat_multidomain_adaptation",
    "seed": 42,
    "device": "cpu",  # auto-detected to cuda if available
    
    # Dataset configurations
    "data": {
        "dataset_type": "synthetic",  # 'synthetic', 'pastis', 'multiearth', 'spacenet'
        "data_dir": "data/sample_dataset",
        "num_samples_per_domain": 120,
        "image_size": 128,
        # Sensor domains (original)
        "sensor_domains": ["sentinel2", "landsat8", "spot6", "sentinel1_sar"],
        # Angle domains (new): nadir + off-nadir acquisition angles over the same scene
        "angle_domains": ["nadir_0", "offnadir_15", "offnadir_30", "oblique_45"],
        # Active domains — set automatically by --mode flag in main.py
        "domains": ["sentinel2", "landsat8", "spot6", "sentinel1_sar"],
        "num_classes": 6,  # [Background, Vegetation, Water, Urban, Agriculture, BareSoil]
        "batch_size": 16,
        "num_workers": 0
    },
    
    # Stage 1: Domain Encoder
    "domain_encoder": {
        "in_channels": 4,  # default RGB+NIR
        "latent_dim": 64,
        "backbone": "resnet_style",
        "lr": 0.0005,
        "epochs": 10,
        "temperature": 0.07,
        "weight_decay": 1e-4
    },
    
    # Stage 2: Unified Multi-Domain Generator & Discriminator
    "generator": {
        "in_channels": 4,
        "out_channels": 4,
        "latent_dim": 64,
        "dim": 32,  # Streamlined filter width for fast CPU execution
        "num_downsample": 2,
        "num_resblocks": 4,
        "lr_g": 0.0003,
        "lr_d": 0.0003,
        "beta1": 0.5,
        "beta2": 0.999,
        "epochs": 12,
        "lambda_adv": 1.0,
        "lambda_sty": 1.0,
        "lambda_cyc": 10.0,
        "lambda_sem": 2.0,  # Semantic / structural preservation loss
        "lambda_id": 1.0,
        "lambda_edge": 2.0, # Sobel boundary edge preservation loss
        "lambda_geo": 5.0   # Geometric preservation loss (active for angle-domain pairs)
    },
    
    # Stage 3: Downstream Segmentation Model
    "downstream": {
        "in_channels": 4,
        "num_classes": 6,
        "lr": 0.001,
        "epochs": 10,
        "model_type": "unet"
    },
    
    # Outputs
    "output_dir": "outputs"
}

def _deep_merge(base: dict, override: dict) -> dict:
    """
    Recursively merges `override` into `base`, in place.
    Nested dicts are merged key-by-key instead of being replaced wholesale, so a user config
    that only sets e.g. data.batch_size doesn't wipe out the other data.* defaults.
    """
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base

def merge_user_config(config: dict, config_path=None) -> dict:
    """
    Merges a user YAML file's overrides into `config` in place (returns config for chaining).

    Callers must apply this AFTER apply_dataset()/apply_mode() — those two unconditionally set
    dataset-specific keys (epochs, batch_size, num_classes, domains, ...), so merging user
    overrides before them would have those overrides silently clobbered right back to defaults.
    """
    if config_path and os.path.exists(config_path):
        with open(config_path, "r") as f:
            user_config = yaml.safe_load(f)
            if user_config:
                _deep_merge(config, user_config)
    return config

def load_config(config_path=None):
    """
    Returns DEFAULT_CONFIG with a user YAML merged on top, with no dataset/mode applied.
    main.py should NOT pass config_path here — see merge_user_config's docstring for why.
    Use load_config(None) to get plain defaults, then apply_dataset()/apply_mode(), then
    merge_user_config() last so user overrides always win.
    """
    config = copy.deepcopy(DEFAULT_CONFIG)
    return merge_user_config(config, config_path)

def apply_mode(config, mode: str):
    """
    Adjusts the active 'domains' list based on the --mode flag.
      'sensor' : original multi-sensor domains only (default)
      'angle'  : multi-angle domains only (nadir / off-nadir)
      'both'   : sensor + angle domains combined in one unified training run
    """
    sensor_domains = config["data"]["sensor_domains"]
    angle_domains  = config["data"]["angle_domains"]
    if mode == "sensor":
        config["data"]["domains"] = sensor_domains
    elif mode == "angle":
        config["data"]["domains"] = angle_domains
    elif mode == "both":
        config["data"]["domains"] = sensor_domains + angle_domains
    else:
        raise ValueError(f"Unknown mode '{mode}'. Choose from: sensor, angle, both")
    return config

def apply_dataset(config, dataset_type: str):
    """
    Configures dataset paths, domains, and class counts for 'synthetic' or 'pastis'.
    """
    if dataset_type == "pastis":
        config["data"]["dataset_type"] = "pastis"
        config["data"]["data_dir"] = "data/france_dataset"
        config["data"]["domains"] = ["sentinel2", "spot6", "sentinel1_sar"]
        config["data"]["sensor_domains"] = ["sentinel2", "spot6", "sentinel1_sar"]
        config["data"]["num_classes"] = 20
        config["downstream"]["num_classes"] = 20
        config["data"]["batch_size"] = 8
        config["domain_encoder"]["epochs"] = 10
        config["generator"]["epochs"] = 12
        config["downstream"]["epochs"] = 10
    elif dataset_type == "synthetic":
        config["data"]["dataset_type"] = "synthetic"
        config["data"]["data_dir"] = "data/sample_dataset"
        config["data"]["domains"] = ["sentinel2", "landsat8", "spot6", "sentinel1_sar"]
        config["data"]["num_classes"] = 6
        config["downstream"]["num_classes"] = 6
    return config

