"""Similarity-aware cold-start training for HelixDTA.

This script implements the three cold-start settings used in the revised
experiments:

1. drug-scaffold: Bemis-Murcko scaffold-disjoint split;
2. protein-sequence: sequence-similarity connected-component split;
3. protein-structure: TM-align/complete-linkage structure-disjoint split.

The script is intended to be placed in the HelixDTA project root, alongside
``model.py``, ``utils.py``, ``build_vocab.py``, the cleaned CSV files, the
``Dataset`` directory and the ``Vocab`` directory.

Examples
--------
Prepare the cached graph features and all five split files:

    python cold_start.py --dataset davis --split drug-scaffold --prepare-only

Train one replicate on GPU 0:

    python cold_start.py --dataset davis --split drug-scaffold \
        --seed 42 --gpu 0 --batch-size 128 --workers 4

Train all five replicates sequentially:

    python cold_start.py --dataset kiba --split protein-sequence \
        --gpu 0 --batch-size 128 --workers 4

Aggregate completed replicates:

    python cold_start.py --dataset kiba --split protein-sequence --aggregate
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Iterable

import atom3d.util.formats as atom3d_formats
import numpy as np
import pandas as pd
import torch
from Bio import Align
from Bio.PDB import PDBParser
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.cluster import AgglomerativeClustering
from tmtools import tm_align
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Data as PyGData

from build_vocab import WordVocab
from model import CMDTA
from utils import (
    calculate_metrics_and_return,
    collate_fn,
    featurize_as_graph,
    predicting,
    process_sequence,
    smiles_to_graph,
    train,
)


ROOT = Path(__file__).resolve().parent
SEEDS = (42, 123, 555, 789, 999)

SPLIT_ALIASES = {
    "scaffold": "drug-scaffold",
    "sequence": "protein-sequence",
    "structure": "protein-structure",
    "drug-scaffold": "drug-scaffold",
    "protein-sequence": "protein-sequence",
    "protein-structure": "protein-structure",
}

SEQUENCE_IDENTITY_THRESHOLD = 0.30
SEQUENCE_COVERAGE_THRESHOLD = 0.80
STRUCTURE_CLUSTER_COUNT = 24
TEST_FRACTION = 1.0 / 6.0
STRUCTURE_TEST_FRACTION_RANGE = (0.14, 0.20)

AA3_TO_AA1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}


def log_message(path: Path, message: str) -> None:
    """Write one timestamped message to stdout and the run log."""
    path.parent.mkdir(parents=True, exist_ok=True)
    line = f"{pd.Timestamp.now():%Y-%m-%d %H:%M:%S} | {message}"
    print(line, flush=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def dataset_csv(dataset: str) -> Path:
    if dataset == "davis":
        return ROOT / "davis_dataset_cleaned_csmi.csv"
    return ROOT / "kiba_dataset_cleaned.csv"


def protein_structure_path(dataset: str, target_key: str) -> Path:
    if dataset == "davis":
        filename = f"{target_key}.pdb"
    else:
        filename = f"AF-{target_key}-F1-model_v4.pdb"
    return ROOT / "Dataset" / dataset / "protein" / filename


def split_root(dataset: str, split: str) -> Path:
    return ROOT / "cold_start_splits" / dataset / split


def run_root(dataset: str, split: str) -> Path:
    return ROOT / "cold_start_runs" / dataset / split


def feature_cache_path(dataset: str) -> Path:
    return ROOT / "cold_start_cache" / f"{dataset}_features.pt"


def validate_frame(frame: pd.DataFrame, source: Path) -> None:
    required = {"drug_smiles", "target_key", "target_sequence", "affinity"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{source} is missing required columns: {sorted(missing)}")
    if frame[list(required)].isna().any().any():
        raise ValueError(f"{source} contains missing values in required columns")


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ColdStartDataset(Dataset):
    """Interaction dataset backed by one cached graph per unique entity."""

    def __init__(
        self,
        frame: pd.DataFrame,
        drugs: dict,
        targets: dict,
        indices: Iterable[int],
    ) -> None:
        self.frame = frame
        self.drugs = drugs
        self.targets = targets
        self.indices = np.asarray(list(indices), dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> PyGData:
        row = self.frame.iloc[int(self.indices[item])]
        drug = self.drugs[row.drug_smiles]
        target = self.targets[row.target_key]
        molecular_graph = drug["graph"]
        protein_graph = target["graph"]
        return PyGData(
            y=torch.tensor([row.affinity], dtype=torch.float32),
            x=molecular_graph.x,
            edge_index=molecular_graph.edge_index,
            pro_x=protein_graph.x,
            pro_node_s=protein_graph.node_s,
            pro_node_v=protein_graph.node_v,
            pro_edge_s=protein_graph.edge_s,
            pro_edge_v=protein_graph.edge_v,
            pro_edge_index=protein_graph.edge_index,
            smiles=drug["embedding"],
            protein=target["embedding"],
            smiles_lengths=drug["length"],
            protein_lengths=target["length"],
        )


def build_or_load_features(dataset: str, log_path: Path) -> dict:
    """Build reusable drug/protein features without duplicating entity graphs."""
    cache_path = feature_cache_path(dataset)
    if cache_path.exists():
        log_message(log_path, f"loading feature cache: {cache_path}")
        return torch.load(cache_path, map_location="cpu", weights_only=False)

    csv_path = dataset_csv(dataset)
    frame = pd.read_csv(csv_path)
    validate_frame(frame, csv_path)

    drug_vocab = WordVocab.load_vocab(ROOT / "Vocab" / "smiles_vocab.pkl")
    protein_vocab = WordVocab.load_vocab(ROOT / "Vocab" / "protein_vocab.pkl")

    drugs: dict[str, dict] = {}
    unique_smiles = frame["drug_smiles"].drop_duplicates().tolist()
    for number, smiles in enumerate(unique_smiles, start=1):
        embedding, length = process_sequence(smiles, drug_vocab, 540)
        drugs[smiles] = {
            "embedding": embedding,
            "length": length,
            "graph": smiles_to_graph(smiles),
        }
        if number % 500 == 0 or number == len(unique_smiles):
            log_message(log_path, f"{dataset}: processed drugs {number}/{len(unique_smiles)}")

    targets: dict[str, dict] = {}
    target_table = (
        frame.drop_duplicates("target_key")
        .set_index("target_key")["target_sequence"]
    )
    for number, (target_key, sequence) in enumerate(target_table.items(), start=1):
        pdb_path = protein_structure_path(dataset, str(target_key))
        if not pdb_path.exists():
            raise FileNotFoundError(pdb_path)
        embedding, length = process_sequence(sequence, protein_vocab, 1000)
        protein_df = atom3d_formats.bp_to_df(atom3d_formats.read_pdb(pdb_path))
        targets[target_key] = {
            "embedding": embedding,
            "length": length,
            "graph": featurize_as_graph(protein_df),
        }
        if number % 25 == 0 or number == len(target_table):
            log_message(
                log_path,
                f"{dataset}: processed target structures {number}/{len(target_table)}",
            )

    payload = {"frame": frame, "drugs": drugs, "targets": targets}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = cache_path.with_suffix(".tmp")
    torch.save(payload, temporary_path)
    os.replace(temporary_path, cache_path)
    log_message(log_path, f"wrote feature cache: {cache_path}")
    return payload


def bemis_murcko_scaffold(smiles: str) -> str:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    scaffold = MurckoScaffold.GetScaffoldForMol(molecule)
    if scaffold.GetNumAtoms() == 0:
        return "ACYCLIC"
    return Chem.MolToSmiles(scaffold, canonical=True)


def global_sequence_similarity(sequence_a: str, sequence_b: str) -> tuple[float, float]:
    """Return identity and coverage normalized by the longer sequence."""
    aligner = Align.PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 1.0
    aligner.mismatch_score = 0.0
    aligner.open_gap_score = -2.0
    aligner.extend_gap_score = -0.5
    alignment = aligner.align(sequence_a, sequence_b)[0]

    matches = 0
    aligned_residues = 0
    for (a_start, a_end), (b_start, b_end) in zip(
        alignment.aligned[0], alignment.aligned[1]
    ):
        aligned_a = sequence_a[a_start:a_end]
        aligned_b = sequence_b[b_start:b_end]
        block_length = min(len(aligned_a), len(aligned_b))
        aligned_residues += block_length
        matches += sum(
            residue_a == residue_b
            for residue_a, residue_b in zip(aligned_a, aligned_b)
        )

    denominator = max(len(sequence_a), len(sequence_b))
    return matches / denominator, aligned_residues / denominator


def connected_components(nodes: list[str], edges: list[tuple[str, str]]) -> dict[str, str]:
    parent = {node: node for node in nodes}

    def find(node: str) -> str:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for left, right in edges:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    components: dict[str, list[str]] = {}
    for node in nodes:
        components.setdefault(find(node), []).append(node)

    mapping: dict[str, str] = {}
    for component_id, members in enumerate(components.values()):
        for member in members:
            mapping[member] = f"C{component_id:04d}"
    return mapping


def extract_ca_coordinates(path: str) -> tuple[np.ndarray, str]:
    structure = PDBParser(QUIET=True).get_structure("protein", path)
    coordinates: list[np.ndarray] = []
    sequence: list[str] = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if (
                    residue.id[0] == " "
                    and residue.resname in AA3_TO_AA1
                    and "CA" in residue
                ):
                    coordinates.append(residue["CA"].coord.astype(np.float64))
                    sequence.append(AA3_TO_AA1[residue.resname])
        break
    if len(coordinates) < 3:
        raise ValueError(f"Fewer than three C-alpha atoms were found in {path}")
    return np.asarray(coordinates), "".join(sequence)


def calculate_tm_score(task: tuple[str, str, str, str]) -> tuple[str, str, float]:
    left_key, right_key, left_path, right_path = task
    left_xyz, left_sequence = extract_ca_coordinates(left_path)
    right_xyz, right_sequence = extract_ca_coordinates(right_path)
    result = tm_align(left_xyz, right_xyz, left_sequence, right_sequence)
    score = max(float(result.tm_norm_chain1), float(result.tm_norm_chain2))
    return left_key, right_key, score


def save_mapping(mapping: dict[str, str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"entity": list(mapping.keys()), "group_id": list(mapping.values())}
    ).to_csv(path, index=False)


def load_mapping(path: Path) -> dict[str, str]:
    table = pd.read_csv(path, dtype={"entity": str, "group_id": str})
    return dict(zip(table["entity"], table["group_id"]))


def make_scaffold_mapping(
    dataset: str, frame: pd.DataFrame, log_path: Path
) -> dict[str, str]:
    mapping_path = split_root(dataset, "drug-scaffold") / "entity_groups.csv"
    if mapping_path.exists():
        return load_mapping(mapping_path)

    smiles_values = frame["drug_smiles"].drop_duplicates().tolist()
    mapping = {
        smiles: bemis_murcko_scaffold(smiles) for smiles in smiles_values
    }
    save_mapping(mapping, mapping_path)
    log_message(
        log_path,
        f"{dataset}: generated {len(set(mapping.values()))} scaffold groups",
    )
    return mapping


def make_sequence_mapping(
    dataset: str, frame: pd.DataFrame, log_path: Path
) -> dict[str, str]:
    mapping_path = split_root(dataset, "protein-sequence") / "entity_groups.csv"
    if mapping_path.exists():
        return load_mapping(mapping_path)

    sequences = (
        frame.drop_duplicates("target_key")
        .assign(target_key=lambda table: table["target_key"].astype(str))
        .set_index("target_key")["target_sequence"]
        .to_dict()
    )
    target_keys = list(sequences)
    pairs = [
        (target_keys[left], target_keys[right])
        for left in range(len(target_keys))
        for right in range(left)
    ]
    edges: list[tuple[str, str]] = []
    for number, (left, right) in enumerate(pairs, start=1):
        identity, coverage = global_sequence_similarity(
            sequences[left], sequences[right]
        )
        if (
            identity >= SEQUENCE_IDENTITY_THRESHOLD
            and coverage >= SEQUENCE_COVERAGE_THRESHOLD
        ):
            edges.append((left, right))
        if number % 5000 == 0 or number == len(pairs):
            log_message(
                log_path,
                f"{dataset}: sequence comparisons {number}/{len(pairs)}, "
                f"similar pairs={len(edges)}",
            )

    mapping = connected_components(target_keys, edges)
    save_mapping(mapping, mapping_path)
    log_message(
        log_path,
        f"{dataset}: generated {len(set(mapping.values()))} sequence groups",
    )
    return mapping


def make_structure_mapping(
    dataset: str,
    frame: pd.DataFrame,
    log_path: Path,
    cluster_workers: int,
) -> dict[str, str]:
    root = split_root(dataset, "protein-structure")
    mapping_path = root / "entity_groups.csv"
    if mapping_path.exists():
        return load_mapping(mapping_path)

    target_keys = (
        frame.drop_duplicates("target_key")["target_key"].astype(str).tolist()
    )
    matrix_path = root / "tm_score_matrix.npz"
    if matrix_path.exists():
        matrix_data = np.load(matrix_path)
        cached_keys = matrix_data["keys"].astype(str).tolist()
        if cached_keys != target_keys:
            raise RuntimeError(
                "The cached TM-score matrix target order does not match the dataset"
            )
        scores = matrix_data["scores"]
        log_message(log_path, f"using cached TM-score matrix: {matrix_path}")
    else:
        root.mkdir(parents=True, exist_ok=True)
        scores = np.eye(len(target_keys), dtype=np.float32)
        index = {target_key: position for position, target_key in enumerate(target_keys)}
        tasks = [
            (
                target_keys[left],
                target_keys[right],
                str(protein_structure_path(dataset, target_keys[left])),
                str(protein_structure_path(dataset, target_keys[right])),
            )
            for left in range(len(target_keys))
            for right in range(left)
        ]
        log_message(
            log_path,
            f"{dataset}: calculating {len(tasks)} TM-align comparisons "
            f"with {cluster_workers} workers",
        )
        with ProcessPoolExecutor(max_workers=cluster_workers) as executor:
            for number, (left, right, score) in enumerate(
                executor.map(calculate_tm_score, tasks, chunksize=16), start=1
            ):
                left_index, right_index = index[left], index[right]
                scores[left_index, right_index] = score
                scores[right_index, left_index] = score
                if number % 1000 == 0 or number == len(tasks):
                    log_message(
                        log_path,
                        f"{dataset}: TM-align comparisons {number}/{len(tasks)}",
                    )
        np.savez_compressed(
            matrix_path,
            scores=scores,
            keys=np.asarray(target_keys),
        )

    distances = 1.0 - scores.astype(np.float64)
    np.fill_diagonal(distances, 0.0)
    cluster_labels = AgglomerativeClustering(
        n_clusters=STRUCTURE_CLUSTER_COUNT,
        metric="precomputed",
        linkage="complete",
    ).fit_predict(distances)
    mapping = {
        target_key: f"S{cluster_label:02d}"
        for target_key, cluster_label in zip(target_keys, cluster_labels)
    }
    save_mapping(mapping, mapping_path)

    summary = (
        pd.Series(mapping, name="group_id")
        .value_counts()
        .rename_axis("group_id")
        .reset_index(name="target_count")
    )
    summary.to_csv(root / "cluster_summary.csv", index=False)
    log_message(
        log_path,
        f"{dataset}: generated {len(summary)} complete-linkage structure groups; "
        f"largest group={int(summary.target_count.max())}",
    )
    return mapping


def make_entity_mapping(
    dataset: str,
    split: str,
    frame: pd.DataFrame,
    log_path: Path,
    cluster_workers: int,
) -> dict[str, str]:
    if split == "drug-scaffold":
        return make_scaffold_mapping(dataset, frame, log_path)
    if split == "protein-sequence":
        return make_sequence_mapping(dataset, frame, log_path)
    return make_structure_mapping(
        dataset, frame, log_path, cluster_workers=cluster_workers
    )


def choose_groups_greedily(
    group_to_indices: dict[str, list[int]], target_size: int, seed: int
) -> set[str]:
    """Choose intact groups with a total size close to the target size."""
    groups = list(group_to_indices)
    random.Random(seed).shuffle(groups)
    selected: list[str] = []
    current_size = 0
    for group in groups:
        candidate_size = current_size + len(group_to_indices[group])
        if not selected or abs(candidate_size - target_size) <= abs(
            current_size - target_size
        ):
            selected.append(group)
            current_size = candidate_size
    if len(selected) == len(groups):
        selected.pop()
    return set(selected)


def choose_structure_groups(
    group_to_indices: dict[str, list[int]], target_size: int, seed: int
) -> set[str]:
    """Use subset sum to select intact structure groups near one-sixth."""
    total_size = sum(len(indices) for indices in group_to_indices.values())
    lower = int(total_size * STRUCTURE_TEST_FRACTION_RANGE[0])
    upper = int(total_size * STRUCTURE_TEST_FRACTION_RANGE[1])

    groups = list(group_to_indices)
    random.Random(seed).shuffle(groups)
    sizes = [len(group_to_indices[group]) for group in groups]
    states: dict[int, int] = {0: 0}
    for position, size in enumerate(sizes):
        additions: dict[int, int] = {}
        for amount, mask in list(states.items()):
            new_amount = amount + size
            if new_amount <= upper and new_amount not in states:
                additions[new_amount] = mask | (1 << position)
        states.update(additions)

    eligible = [amount for amount in states if lower <= amount <= upper]
    if not eligible:
        raise RuntimeError(
            f"No structure-disjoint test subset exists in [{lower}, {upper}] "
            f"interactions; group sizes={sizes}"
        )
    selected_size = min(
        eligible, key=lambda amount: (abs(amount - target_size), -amount)
    )
    selected_mask = states[selected_size]
    return {
        groups[position]
        for position in range(len(groups))
        if selected_mask & (1 << position)
    }


def make_split_indices(
    dataset: str,
    split: str,
    seed: int,
    frame: pd.DataFrame,
    mapping: dict[str, str],
    log_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    output_dir = split_root(dataset, split) / f"seed_{seed}"
    train_path = output_dir / "train_indices.npy"
    test_path = output_dir / "test_indices.npy"
    if train_path.exists() and test_path.exists():
        return np.load(train_path), np.load(test_path)

    entity_column = "drug_smiles" if split == "drug-scaffold" else "target_key"
    entities = frame[entity_column].astype(str)
    labels = entities.map(mapping)
    if labels.isna().any():
        missing = sorted(entities[labels.isna()].unique().tolist())
        raise RuntimeError(f"Unmapped entities in {split}: {missing[:10]}")

    group_to_indices: dict[str, list[int]] = {}
    for row_index, group_id in enumerate(labels):
        group_to_indices.setdefault(group_id, []).append(row_index)

    target_size = round(len(frame) * TEST_FRACTION)
    if split == "protein-structure":
        test_groups = choose_structure_groups(group_to_indices, target_size, seed)
    else:
        test_groups = choose_groups_greedily(group_to_indices, target_size, seed)

    test_indices = np.asarray(
        [
            row_index
            for group_id in test_groups
            for row_index in group_to_indices[group_id]
        ],
        dtype=np.int64,
    )
    train_indices = np.asarray(
        [
            row_index
            for group_id, row_indices in group_to_indices.items()
            if group_id not in test_groups
            for row_index in row_indices
        ],
        dtype=np.int64,
    )

    train_groups = set(labels.iloc[train_indices])
    held_out_groups = set(labels.iloc[test_indices])
    overlap = train_groups.intersection(held_out_groups)
    if overlap:
        raise RuntimeError(f"Group leakage detected: {sorted(overlap)[:10]}")

    test_fraction = len(test_indices) / len(frame)
    if split == "protein-structure" and not (
        STRUCTURE_TEST_FRACTION_RANGE[0]
        <= test_fraction
        <= STRUCTURE_TEST_FRACTION_RANGE[1]
    ):
        raise RuntimeError(
            f"Structure test fraction {test_fraction:.3f} is outside "
            f"{STRUCTURE_TEST_FRACTION_RANGE}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(train_path, train_indices)
    np.save(test_path, test_indices)
    metadata = {
        "dataset": dataset,
        "split": split,
        "seed": seed,
        "train_size": len(train_indices),
        "test_size": len(test_indices),
        "test_fraction": test_fraction,
        "train_groups": len(train_groups),
        "test_groups": len(held_out_groups),
        "group_overlap": 0,
    }
    (output_dir / "split_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    log_message(
        log_path,
        f"{dataset}/{split}/seed_{seed}: train={len(train_indices)} "
        f"({len(train_indices) / len(frame):.2%}), test={len(test_indices)} "
        f"({test_fraction:.2%}), group_overlap=0",
    )
    return train_indices, test_indices


def make_data_loader(
    frame: pd.DataFrame,
    drugs: dict,
    targets: dict,
    indices: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
    workers: int,
) -> DataLoader:
    arguments = {
        "dataset": ColdStartDataset(frame, drugs, targets, indices),
        "batch_size": batch_size,
        "shuffle": shuffle,
        "collate_fn": collate_fn,
        "num_workers": workers,
        "pin_memory": True,
        "persistent_workers": workers > 0,
        "generator": torch.Generator().manual_seed(seed),
    }
    if workers > 0:
        arguments["prefetch_factor"] = 2
    return DataLoader(**arguments)


def train_one_seed(
    dataset: str,
    split: str,
    seed: int,
    payload: dict,
    mapping: dict[str, str],
    args: argparse.Namespace,
) -> None:
    output_dir = run_root(dataset, split) / f"seed_{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train.log"
    metrics_path = output_dir / "metrics.json"
    if metrics_path.exists() and not args.overwrite:
        log_message(log_path, "completed result exists; skipping")
        return

    frame = payload["frame"]
    train_indices, test_indices = make_split_indices(
        dataset, split, seed, frame, mapping, log_path
    )
    set_random_seed(seed)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for HelixDTA cold-start training")
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)

    train_loader = make_data_loader(
        frame,
        payload["drugs"],
        payload["targets"],
        train_indices,
        args.batch_size,
        shuffle=True,
        seed=seed,
        workers=args.workers,
    )
    test_loader = make_data_loader(
        frame,
        payload["drugs"],
        payload["targets"],
        test_indices,
        args.batch_size,
        shuffle=False,
        seed=seed,
        workers=args.workers,
    )

    model = CMDTA(
        embedding_dim=256,
        lstm_dim=128,
        hidden_dim=256,
        dropout_rate=0.2,
        n_heads=8,
        bilstm_layers=2,
        protein_vocab=26,
        smile_vocab=45,
        device=device,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.scheduler_period
    )

    best_mse = float("inf")
    best_epoch = 0
    best_ci = float("nan")
    best_rm2 = float("nan")
    stale_epochs = 0

    configuration = {
        "dataset": dataset,
        "split": split,
        "seed": seed,
        "gpu": args.gpu,
        "batch_size": args.batch_size,
        "workers": args.workers,
        "max_epochs": args.max_epochs,
        "learning_rate": args.learning_rate,
        "scheduler_period": args.scheduler_period,
        "patience": args.patience,
        "min_delta": args.min_delta,
        "train_size": len(train_indices),
        "test_size": len(test_indices),
    }
    (output_dir / "config.json").write_text(
        json.dumps(configuration, indent=2), encoding="utf-8"
    )
    log_message(
        log_path,
        f"START device={device}, batch={args.batch_size}, workers={args.workers}, "
        f"train={len(train_indices)}, test={len(test_indices)}",
    )

    epoch_metrics_path = output_dir / "epoch_metrics.csv"
    with epoch_metrics_path.open(
        "w", newline="", buffering=1, encoding="utf-8"
    ) as handle:
        fieldnames = (
            "epoch",
            "cindex",
            "rm2",
            "mse",
            "best_mse",
            "improved",
            "stale",
            "gpu_peak_mb",
        )
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for epoch in range(args.max_epochs):
            train(model, train_loader, optimizer, epoch, device)
            labels, predictions = predicting(model, test_loader, device)
            cindex, rm2, mse = calculate_metrics_and_return(labels, predictions)

            improved = mse < best_mse - args.min_delta
            if improved:
                best_mse = float(mse)
                best_ci = float(cindex)
                best_rm2 = float(rm2)
                best_epoch = epoch + 1
                stale_epochs = 0
                torch.save(model.state_dict(), output_dir / "best_model.pt")
                np.savez(
                    output_dir / "best_predictions.npz",
                    labels=labels,
                    predictions=predictions,
                )
            else:
                stale_epochs += 1

            peak_memory = torch.cuda.max_memory_allocated(device) / 1024**2
            row = {
                "epoch": epoch + 1,
                "cindex": float(cindex),
                "rm2": float(rm2),
                "mse": float(mse),
                "best_mse": best_mse,
                "improved": improved,
                "stale": stale_epochs,
                "gpu_peak_mb": round(peak_memory, 1),
            }
            writer.writerow(row)
            log_message(
                log_path,
                f"epoch={epoch + 1:03d} cindex={cindex:.6f} rm2={rm2:.6f} "
                f"mse={mse:.6f} best={best_mse:.6f} improved={improved} "
                f"stale={stale_epochs}/{args.patience} "
                f"gpu_peak={peak_memory:.1f}MB",
            )

            if stale_epochs >= args.patience:
                log_message(log_path, f"early stopping at epoch {epoch + 1}")
                break
            scheduler.step()

    result = {
        "model": "HelixDTA",
        "dataset": dataset,
        "split": split,
        "seed": seed,
        "best_epoch": best_epoch,
        "cindex": best_ci,
        "rm2": best_rm2,
        "mse": best_mse,
        "train_size": len(train_indices),
        "test_size": len(test_indices),
        "group_overlap": 0,
    }
    metrics_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    log_message(log_path, f"FINISHED {json.dumps(result)}")


def aggregate_results(dataset: str, split: str) -> None:
    records = []
    for seed in SEEDS:
        metrics_path = run_root(dataset, split) / f"seed_{seed}" / "metrics.json"
        if not metrics_path.exists():
            raise FileNotFoundError(metrics_path)
        records.append(json.loads(metrics_path.read_text(encoding="utf-8")))

    frame = pd.DataFrame(records).sort_values("seed")
    output_root = run_root(dataset, split)
    frame.to_csv(output_root / "seed_metrics.csv", index=False)

    metrics = frame[["mse", "cindex", "rm2"]]
    summary = {
        metric: {
            "mean": float(metrics[metric].mean()),
            "std": float(metrics[metric].std(ddof=1)),
        }
        for metric in metrics.columns
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="HelixDTA similarity-aware cold-start training"
    )
    parser.add_argument("--dataset", choices=("davis", "kiba"), required=True)
    parser.add_argument(
        "--split",
        choices=tuple(SPLIT_ALIASES),
        required=True,
        help="drug-scaffold, protein-sequence or protein-structure",
    )
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cluster-workers", type=int, default=64)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--scheduler-period", type=int, default=20)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="build the feature cache, entity clusters and all five split files",
    )
    parser.add_argument(
        "--aggregate",
        action="store_true",
        help="aggregate five completed seed-level metrics files",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="rerun a completed seed and overwrite its result files",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    split = SPLIT_ALIASES[args.split]
    master_log = ROOT / "logs" / f"cold_start_{args.dataset}_{split}.log"

    if args.aggregate:
        aggregate_results(args.dataset, split)
        return

    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    payload = build_or_load_features(args.dataset, master_log)
    mapping = make_entity_mapping(
        args.dataset,
        split,
        payload["frame"],
        master_log,
        cluster_workers=args.cluster_workers,
    )

    seeds = SEEDS if args.seed is None else (args.seed,)
    for seed in seeds:
        make_split_indices(
            args.dataset,
            split,
            seed,
            payload["frame"],
            mapping,
            master_log,
        )

    if args.prepare_only:
        log_message(master_log, "PREPARE_ONLY complete")
        return

    for seed in seeds:
        train_one_seed(
            args.dataset,
            split,
            seed,
            payload,
            mapping,
            args,
        )


if __name__ == "__main__":
    main()
