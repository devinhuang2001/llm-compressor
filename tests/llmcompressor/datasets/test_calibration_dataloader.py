import pytest
from datasets import Dataset

from llmcompressor.args import DatasetArguments
from llmcompressor.datasets.utils import format_calibration_data


@pytest.mark.unit
@pytest.mark.parametrize("shuffle", [False, True])
@pytest.mark.parametrize(
    "num_samples, expected_partitions",
    [
        (None, [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]),
        (8, [[0, 1, 2, 3], [4, 5, 6, 7]]),
        (9, [[0, 1, 2, 3], [4, 5, 6, 7, 8]]),
    ],
)
def test_calibration_dataloader_reads_rank_partition(
    monkeypatch, shuffle, num_samples, expected_partitions
):
    dataset = Dataset.from_dict({"input_ids": [[i] * (i + 1) for i in range(10)]})
    args = DatasetArguments(
        num_calibration_samples=num_samples,
        shuffle_calibration_samples=shuffle,
        batch_size=1,
    )
    monkeypatch.setattr(
        "llmcompressor.datasets.utils._is_dist_and_same_ds", lambda dataset: True
    )
    monkeypatch.setattr("llmcompressor.datasets.utils.dist.get_world_size", lambda: 2)

    for rank, expected_rows in enumerate(expected_partitions):
        monkeypatch.setattr("llmcompressor.datasets.utils.dist.get_rank", lambda: rank)
        dataloader = format_calibration_data(args, dataset, processor=None)

        # Read actual batches: sampler indices alone cannot detect a DataLoader
        # indexing the full dataset instead of this rank's partition.
        rows = sorted(int(batch["input_ids"][0, 0]) for batch in dataloader)
        assert rows == expected_rows


@pytest.mark.unit
@pytest.mark.parametrize("shuffle", [False, True])
def test_calibration_dataloader_without_automatic_partition(monkeypatch, shuffle):
    dataset = Dataset.from_dict({"input_ids": [[i] for i in [5, 6, 7, 8, 9]]})
    args = DatasetArguments(
        num_calibration_samples=None,
        shuffle_calibration_samples=shuffle,
        batch_size=1,
    )
    # Covers both a single process and ranks given different datasets.
    monkeypatch.setattr(
        "llmcompressor.datasets.utils._is_dist_and_same_ds", lambda dataset: False
    )

    dataloader = format_calibration_data(args, dataset, processor=None)

    assert sorted(int(batch["input_ids"][0, 0]) for batch in dataloader) == [
        5,
        6,
        7,
        8,
        9,
    ]
