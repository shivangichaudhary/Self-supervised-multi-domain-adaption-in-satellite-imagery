import yaml

from configs.config import load_config, apply_mode, apply_dataset, DEFAULT_CONFIG


def test_load_config_no_path_returns_defaults():
    config = load_config(None)
    assert config["data"]["domains"] == DEFAULT_CONFIG["data"]["domains"]
    assert config["generator"]["epochs"] == DEFAULT_CONFIG["generator"]["epochs"]


def test_user_config_deep_merges_nested_dict(tmp_path):
    """
    Regression test: overriding one nested key (data.batch_size) must not wipe out the
    other defaults under data.* (domains, image_size, ...) the way a shallow dict.update
    would.
    """
    cfg_path = tmp_path / "custom.yaml"
    cfg_path.write_text(yaml.dump({"data": {"batch_size": 4}}))

    config = load_config(str(cfg_path))
    assert config["data"]["batch_size"] == 4
    assert config["data"]["domains"] == DEFAULT_CONFIG["data"]["domains"]
    assert config["data"]["image_size"] == DEFAULT_CONFIG["data"]["image_size"]
    assert config["data"]["num_samples_per_domain"] == DEFAULT_CONFIG["data"]["num_samples_per_domain"]


def test_apply_mode_both_combines_domain_lists():
    config = load_config(None)
    config = apply_mode(config, "both")
    assert set(config["data"]["domains"]) == set(
        DEFAULT_CONFIG["data"]["sensor_domains"] + DEFAULT_CONFIG["data"]["angle_domains"]
    )


def test_apply_dataset_pastis_sets_expected_domains():
    config = load_config(None)
    config = apply_dataset(config, "pastis")
    assert config["data"]["domains"] == ["sentinel2", "spot6", "sentinel1_sar"]
    assert config["data"]["num_classes"] == 20
    assert config["downstream"]["num_classes"] == 20
