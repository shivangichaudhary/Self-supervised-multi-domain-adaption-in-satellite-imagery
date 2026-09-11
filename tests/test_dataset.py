from data.dataset import MultiSensorSatelliteDataset
from data.synthetic_generator import SyntheticSatelliteGenerator


def _build_tiny_dataset(tmp_path, domains):
    gen = SyntheticSatelliteGenerator(image_size=16, seed=0)
    gen.generate_dataset(output_dir=str(tmp_path), num_samples_per_domain=4, domains=domains)
    return tmp_path


def test_sensor_domain_samples_get_no_angle_sentinel(tmp_path):
    data_dir = _build_tiny_dataset(tmp_path, ["sentinel2", "landsat8"])
    ds = MultiSensorSatelliteDataset(root_dir=str(data_dir), domains=["sentinel2", "landsat8"], split="train")
    sample = ds[0]
    assert sample["src_angle_deg"] == -1.0


def test_angle_domain_samples_get_real_angle_degree(tmp_path):
    data_dir = _build_tiny_dataset(tmp_path, ["nadir_0", "offnadir_30"])
    ds = MultiSensorSatelliteDataset(root_dir=str(data_dir), domains=["nadir_0", "offnadir_30"], split="train")

    seen_angles = set()
    for i in range(len(ds)):
        sample = ds[i]
        assert sample["src_angle_deg"] >= 0.0
        seen_angles.add(sample["src_angle_deg"])
    assert seen_angles <= {0.0, 30.0}


def test_mixed_mode_batch_has_both_sentinel_and_real_angles(tmp_path):
    """
    Sanity check for --mode both: a dataset spanning sensor + angle domains must expose
    enough signal (mix of -1 sentinels and real degrees across src/tgt) for train_generator's
    per-sample geo_mask to actually do something.
    """
    domains = ["sentinel2", "offnadir_15"]
    data_dir = _build_tiny_dataset(tmp_path, domains)
    ds = MultiSensorSatelliteDataset(root_dir=str(data_dir), domains=domains, split="train")

    src_angles = [ds[i]["src_angle_deg"] for i in range(len(ds))]
    assert -1.0 in src_angles  # sentinel2 samples
    assert 15.0 in src_angles  # offnadir_15 samples
