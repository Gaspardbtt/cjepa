"""Prétraiter CLEVRER, puis afficher la grille de seuils de couverture.

Depuis la racine du dépôt :
    python -m probing.clevrer.calibrate_matching --paths configs/paths.fir.yaml

Même commande pour reprendre : les vidéos déjà calculées sont réutilisées.
Pour changer les seuils sans charger les slots ni le décodeur :
    python -m probing.clevrer.calibrate_matching --summary-only

Les masques restent en mémoire le temps d'un petit batch. Seules les matrices
et les statistiques utiles sont conservées, pas les grandes images binaires.
Les seuils filtrent des FENÊTRES de collision, pas des vidéos entières.
"""

import argparse
import json
import os
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from probing.clevrer import match_objets as mo


ROOT = Path(__file__).resolve().parents[2]
CACHE_VERSION = 1


def resolve_path(value):
    path = Path(os.path.expandvars(str(value))).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def read_json(path):
    with Path(path).open() as f:
        return json.load(f)


def write_json(path, data):
    """Un fichier temporaire évite de garder une vidéo à moitié écrite."""
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, allow_nan=False)
    temporary.replace(path)


def select_collision_windows(annotation, n_frames, config):
    """Sélection structurelle par annotations uniquement, avant les seuils."""
    windows, exclusions = [], []
    trajectory = {f["frame_id"]: f for f in annotation["motion_trajectory"]}
    events = annotation["collision"]
    for event_index, event in enumerate(events):
        collision = int(event["frame_id"])
        participants = [int(obj) for obj in event["object_ids"]]
        last_observed = collision - 1
        first_observed = last_observed - (config["history_size"] - 1) * config["frame_step"]
        history = list(range(first_observed, last_observed + 1, config["frame_step"]))
        target, prediction_step = last_observed + config["frame_step"], 1
        while target <= collision:
            target += config["frame_step"]
            prediction_step += 1
        frames = history + [target]
        reason = None
        if min(frames) < 0 or target >= n_frames or any(t not in trajectory for t in frames):
            reason = "window_out_of_bounds"
        elif prediction_step > config["num_predictions"]:
            reason = "prediction_horizon_too_short"
        else:
            visible = [
                {o["object_id"] for o in trajectory[t]["objects"] if o["inside_camera_view"]}
                for t in frames
            ]
            required = set.intersection(*visible)
            if not set(participants) <= required:
                reason = "participant_not_always_visible"
            elif len(required) < config["min_visible_objects"]:
                reason = "not_enough_visible_objects"
            elif any(
                j != event_index and first_observed <= other["frame_id"] <= target
                and set(other["object_ids"]) & set(participants)
                for j, other in enumerate(events)
            ):
                # Cela exclut aussi une autre collision simultanée impliquant un participant.
                reason = "another_collision_in_window"
        if reason:
            exclusions.append({"event_index": event_index, "collision_frame": collision, "reason": reason})
            continue
        windows.append({
            "event_index": event_index, "collision_frame": collision,
            "collision_objects": participants, "history_frames": history,
            "target_frame": target, "prediction_step": prediction_step,
            "frames": frames, "required_object_ids": sorted(required),
        })
    return windows, exclusions


def compute_frame_metrics(decoder, slots, proposals, annotation, frame_ids, batch_size):
    """Décode seulement les images utiles, une fois par vidéo, en petits batchs."""
    proposal_frames = {f["frame_index"]: f for f in proposals["frames"]}
    available = [t for t in frame_ids if t in proposal_frames and proposal_frames[t]["objects"]]
    results = {}
    for start in range(0, len(available), batch_size):
        indices = available[start:start + batch_size]
        size = tuple(proposal_frames[indices[0]]["objects"][0]["mask"]["size"])
        slot_masks = mo.decode_slot_masks(decoder, slots[indices], size)
        for i, frame_index in enumerate(indices):
            object_masks, objects, actual_size = mo.decode_object_masks(
                proposal_frames[frame_index], annotation
            )
            if actual_size != size:
                raise ValueError("Les masques d'une vidéo n'ont pas tous la même résolution.")
            results[frame_index] = {
                "objects": objects,
                "iou": mo.compute_iou_matrix(slot_masks[i], object_masks),
                "coverage": mo.compute_coverage_matrix(slot_masks[i], object_masks),
            }
    return results


def compute_window_metrics(window, frame_results):
    """Résume une fenêtre sans choisir de seuil : un minimum et un maximum."""
    record = dict(window)
    record.update(reference_complete=False, identities_stable=False,
                  measurements_complete=False, min_target_coverage=None,
                  max_other_coverage=None, slot_to_object={}, pair_metrics=[])
    missing = [t for t in window["frames"] if t not in frame_results]
    if missing:
        record.update(status="missing_proposal_frames", missing_frames=missing)
        return record

    # Les associations brutes restent indépendantes des seuils.
    mappings = []
    for t in window["frames"]:
        result = frame_results[t]
        mapping = {}
        for slot, column in mo.assign_slots(result["iou"]):
            object_id = result["objects"][column]["object_id"]
            if object_id is not None:
                mapping[int(slot)] = int(object_id)
        mappings.append(mapping)
    reference = {
        slot: obj for slot, obj in mappings[0].items()
        if obj in window["required_object_ids"]
    }
    record["slot_to_object"] = reference
    record["reference_complete"] = set(reference.values()) == set(window["required_object_ids"])
    record["identities_stable"] = mo.check_temporal_consistency(reference, mappings)

    for slot, obj in reference.items():
        targets, others = [], []
        for t in window["frames"]:
            result = frame_results[t]
            columns = [j for j, o in enumerate(result["objects"]) if o["object_id"] == obj]
            if len(columns) == 1:
                target, other = mo.get_pair_coverage(result["coverage"], slot, columns[0])
                targets.append(target)
                others.append(other)
        complete = len(targets) == len(window["frames"])
        record["pair_metrics"].append({
            "slot": slot, "object_id": obj, "measurements_complete": complete,
            "min_target_coverage": min(targets) if complete else None,
            "max_other_coverage": max(others) if complete else None,
            "identity_stable": all(m.get(slot) == obj for m in mappings),
        })
    pairs = record["pair_metrics"]
    record["measurements_complete"] = bool(pairs) and all(p["measurements_complete"] for p in pairs)
    if record["measurements_complete"]:
        record["min_target_coverage"] = min(p["min_target_coverage"] for p in pairs)
        record["max_other_coverage"] = max(p["max_other_coverage"] for p in pairs)
    record["status"] = "measured"
    return record


def process_video(video_id, video_slots, annotation_path, proposal_path, decoder, config):
    annotation, proposals = read_json(annotation_path), read_json(proposal_path)
    if annotation["scene_index"] != video_id or proposals["video_index"] != video_id:
        raise ValueError("Identifiants de vidéo incohérents dans les JSON.")
    slots = np.asarray(video_slots, dtype=np.float32)
    if slots.ndim != 3 or slots.shape[0] != len(annotation["motion_trajectory"]):
        raise ValueError("Slots attendus : [T,S,D], avec une entrée par image brute.")
    windows, exclusions = select_collision_windows(annotation, len(slots), config)
    frame_ids = sorted({t for window in windows for t in window["frames"]})
    frame_results = compute_frame_metrics(
        decoder, slots, proposals, annotation, frame_ids, config["batch_frames"]
    )
    return {
        "video_id": video_id, "status": "ok", "n_collisions": len(annotation["collision"]),
        "windows": [compute_window_metrics(window, frame_results) for window in windows],
        "exclusions": exclusions,
    }


def summarize(output_dir, config, video_ids):
    """Relit les petits fichiers calculés et imprime la matrice globale."""
    windows, exclusions, video_status = [], [], Counter()
    for video_id in video_ids:
        cache_path = output_dir / "videos" / f"video_{video_id:05d}.json"
        if not cache_path.exists():
            video_status["pending"] += 1
            continue
        result = read_json(cache_path)
        video_status[result["status"]] += 1
        for record in result["windows"]:
            windows.append({"video_id": video_id, **record})
        for record in result["exclusions"]:
            exclusions.append({"video_id": video_id, **record})
    with (output_dir / "windows.jsonl").open("w") as f:
        for window in windows:
            f.write(json.dumps(window, ensure_ascii=False, allow_nan=False) + "\n")
    pd.DataFrame(exclusions, columns=["video_id", "event_index", "collision_frame", "reason"]).to_csv(
        output_dir / "exclusions.csv", index=False
    )

    rows = []
    for min_target in config["thresholds"]["min_target_coverage"]:
        for max_other in config["thresholds"]["max_other_coverage"]:
            accepted = [w for w in windows
                        if w["reference_complete"] and w["identities_stable"]
                        and w["measurements_complete"]
                        and w["min_target_coverage"] >= min_target
                        and w["max_other_coverage"] <= max_other]
            rows.append({
                "min_target_coverage": min_target, "max_other_coverage": max_other,
                "accepted_windows": len(accepted),
                "accepted_videos": len({w["video_id"] for w in accepted}),
                "eligible_windows": len(windows),
                "retained_fraction": len(accepted) / len(windows) if windows else 0.0,
            })
    table = pd.DataFrame(rows)
    matrix = table.pivot(index="min_target_coverage", columns="max_other_coverage",
                         values="accepted_windows")
    table.to_csv(output_dir / "thresholds.csv", index=False)
    matrix.to_csv(output_dir / "threshold_matrix.csv")
    summary = {"requested_videos": len(video_ids), "video_status": dict(video_status),
               "eligible_windows": len(windows), "excluded_records": len(exclusions),
               "window_status": dict(Counter(w["status"] for w in windows))}
    write_json(output_dir / "summary.json", summary)
    print("\nVidéos :", dict(video_status), "| demandées :", len(video_ids))
    print("Fenêtres structurellement éligibles :", len(windows))
    print("Lignes = couverture minimale de l'objet associé.")
    print("Colonnes = couverture maximale permise de chaque autre objet.")
    print("Valeurs = nombres de fenêtres complètes et stables conservées.\n")
    print(matrix.to_string())

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4))
    values = matrix.to_numpy()
    image = ax.imshow(values, vmin=0, vmax=max(1, int(values.max())), cmap="YlGnBu", aspect="auto")
    ax.set_xticks(range(len(matrix.columns)), labels=[f"{v:.0%}" for v in matrix.columns])
    ax.set_yticks(range(len(matrix.index)), labels=[f"{v:.0%}" for v in matrix.index])
    for i in range(len(matrix.index)):
        for j in range(len(matrix.columns)):
            ax.text(j, i, str(values[i, j]), ha="center", va="center",
                    color="white" if values[i, j] > values.max() / 2 else "black")
    ax.set_xlabel("Couverture maximale des autres objets")
    ax.set_ylabel("Couverture minimale de l'objet associé")
    ax.set_title(f"{config['split']} : fenêtres conservées sur {len(windows)} éligibles")
    fig.colorbar(image, ax=ax, label="Fenêtres conservées")
    fig.tight_layout()
    fig.savefig(output_dir / "threshold_matrix.png", dpi=160)
    plt.close(fig)
    print("\nRésultats :", output_dir)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/calibrate_matching.yaml")
    parser.add_argument("--paths", default="configs/paths.fir.yaml")
    parser.add_argument("--output-dir")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--split", choices=["train", "val"])
    parser.add_argument("--max-videos", type=int, help="Petit essai ; omettre pour tout le split.")
    parser.add_argument("--summary-only", action="store_true", help="Aucun chargement de slots ou de poids.")
    args = parser.parse_args()
    config = OmegaConf.to_container(OmegaConf.load(resolve_path(args.config)), resolve=True)
    paths = OmegaConf.to_container(OmegaConf.load(resolve_path(args.paths)), resolve=True)["paths"]
    if args.split:
        config["split"] = args.split
    if args.device:
        config["device"] = args.device
    output_dir = resolve_path(args.output_dir or paths["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "run_config.json"
    if args.summary_only:
        manifest = read_json(manifest_path)
        config["split"] = manifest["signature"]["settings"]["split"]
        summarize(output_dir, config, manifest["video_ids"])
        return

    for name in ["history_size", "num_predictions", "frame_step", "min_visible_objects", "batch_frames"]:
        if config[name] < 1:
            raise ValueError(f"{name} doit être positif.")
    if config["min_visible_objects"] < 3:
        raise ValueError("Il faut au moins trois objets visibles pour ce diagnostic.")
    if args.max_videos is not None and args.max_videos < 1:
        raise ValueError("--max-videos doit être positif.")
    for values in config["thresholds"].values():
        if not values or any(not 0 <= v <= 1 for v in values):
            raise ValueError("La grille de seuils doit contenir des fractions entre 0 et 1.")
    paths = {name: resolve_path(paths[name]) for name in
             ["slots", "encoder_checkpoint", "annotations_dir", "object_masks_dir"]}
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    settings = {k: v for k, v in config.items() if k not in ("thresholds", "device")}
    signature = {"cache_version": CACHE_VERSION, "settings": settings,
                 "paths": {k: str(v) for k, v in paths.items()},
                 "input_files": {k: {"size": paths[k].stat().st_size,
                                      "mtime_ns": paths[k].stat().st_mtime_ns}
                                 for k in ["slots", "encoder_checkpoint"]}}
    if manifest_path.exists() and read_json(manifest_path)["signature"] != signature:
        raise ValueError("Paramètres/entrées différents du cache. Choisir un nouveau --output-dir.")

    print("Chargement unique des slots :", paths["slots"], flush=True)
    with paths["slots"].open("rb") as f:
        all_slots = pickle.load(f)
    split_slots = all_slots[config["split"]]
    del all_slots  # libère les autres splits après le pic de mémoire du pickle
    # Le val officiel contient 5 000 vidéos. Les slots manquants sont signalés.
    video_ids = list(range(10000, 15000)) if config["split"] == "val" else sorted(
        int(Path(key).stem.split("_")[0]) for key in split_slots
    )
    if args.max_videos:
        video_ids = video_ids[:args.max_videos]
    write_json(manifest_path, {"signature": signature, "video_ids": video_ids})
    print("Split :", config["split"], "| vidéos demandées :", len(video_ids), flush=True)
    annotation_files = {int(p.stem.split("_")[-1]): p
                        for p in paths["annotations_dir"].rglob("annotation_*.json")}
    proposal_files = {int(p.stem.split("_")[-1]): p
                      for p in paths["object_masks_dir"].rglob("proposal_*.json")}
    device = config["device"]
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    decoder = mo.load_decoder(paths["encoder_checkpoint"], resolve_path(config["decoder_config"]), device)
    cache_dir = output_dir / "videos"
    cache_dir.mkdir(exist_ok=True)
    for index, video_id in enumerate(video_ids):
        cache_path = cache_dir / f"video_{video_id:05d}.json"
        # Les vidéos en erreur sont retentées ; seules les vidéos terminées sont sautées.
        if cache_path.exists() and read_json(cache_path)["status"] == "ok":
            status = "cached"
        else:
            try:
                result = process_video(
                    video_id, split_slots[f"{video_id}_pixels.mp4"],
                    annotation_files[video_id], proposal_files[video_id], decoder, config
                )
            except (KeyError, FileNotFoundError, ValueError) as error:
                result = {"video_id": video_id, "status": "data_error", "windows": [],
                          "exclusions": [{"event_index": None, "collision_frame": None,
                                          "reason": f"{type(error).__name__}: {error}"}]}
                print(f"Vidéo {video_id} : données indisponibles/invalides : {error}", flush=True)
            write_json(cache_path, result)
            status = result["status"]
        if index == 0 or (index + 1) % 25 == 0 or index + 1 == len(video_ids):
            print(f"[{index + 1}/{len(video_ids)}] vidéo {video_id} : {status}", flush=True)
    summary = summarize(output_dir, config, video_ids)
    if not summary["video_status"].get("ok", 0):
        raise RuntimeError("Aucune vidéo traitée correctement. Vérifier les chemins et les données.")


if __name__ == "__main__":
    main()
