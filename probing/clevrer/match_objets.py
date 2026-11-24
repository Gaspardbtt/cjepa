"""Associer les slots VideoSAUR aux objets CLEVRER avec des mesures lisibles.

L'IoU propose une affectation un à un. Pour accepter un couple, on exige :
- une fraction suffisante de l'objet associé couverte par le slot ;
- une faible fraction couverte de chacun des autres objets.
On vérifie ensuite que les identités restent stables dans la fenêtre.

Exemple dans play.ipynb (lancé depuis la racine du dépôt) :

    from probing.clevrer import match_objets as mo

    z = mo.load_video_slots(
        "checkpoints/clevrer_videosaur_slots_preview.pkl", "0_pixels.mp4"
    )
    decoder = mo.load_decoder("checkpoints/clevrer_videosaur_model.ckpt")
    result = mo.prepare_frame(
        decoder, z, frame_index=18,
        proposal_path="checkpoints/derender_proposals/proposal_00000.json",
        annotation_path=("checkpoints/annotation_train/"
                         "annotation_00000-01000/annotation_00000.json"),
    )
    print(result["iou"])
    mo.show_pair(result, slot_index=0, proposal_index=0)

Les indices de colonnes de la matrice IoU sont des indices de PROPOSITIONS,
pas des object_id. Voir result["objects"][colonne]["object_id"].
"""

import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


DEFAULT_CONFIG = (
    Path(__file__).resolve().parents[2]
    / "src/third_party/videosaur/configs/videosaur/clevrer_dinov2_hf.yml"
)


def find_video_path(videos_dir, video_id):
    """Retrouve video_XXXXX.mp4, y compris dans video_10000-11000, etc.

    Exemple : find_video_path('/scratch/gaps/clevrer_val', 10000).
    Le prétraitement des slots n'a pas besoin des MP4 ; ceci sert aux contrôles visuels.
    """
    filename = f"video_{int(video_id):05d}.mp4"
    matches = sorted(Path(videos_dir).rglob(filename))
    if not matches:
        raise FileNotFoundError(f"{filename} introuvable sous {videos_dir}")
    if len(matches) > 1:
        raise ValueError(f"Plusieurs fichiers {filename} trouvés : {matches}")
    return matches[0]


def load_video_slots(slot_path, video_id, split="train"):
    """Retourne [T, S, D]. Utiliser le petit aperçu sur le Mac !

    Attention : pickle.load charge TOUT le fichier avant de choisir une vidéo.
    """
    with Path(slot_path).open("rb") as f:
        data = pickle.load(f)
    slots = np.asarray(data[split][video_id], dtype=np.float32)
    if slots.ndim != 3:
        raise ValueError(f"Slots attendus : [T, S, D], reçu : {slots.shape}")
    return slots


def load_decoder(checkpoint_path, config_path=DEFAULT_CONFIG, device="cpu"):
    """Charge uniquement le décodeur entraîné, sans construire DINO.

    Le checkpoint doit être celui de l'encodeur ayant produit les slots.
    """
    from src.third_party.videosaur.videosaur import configuration, modules

    config = configuration.load_config(Path(config_path))
    decoder = modules.build_decoder(config.model.decoder)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    prefix = "decoder.module."  # MapOverTime dans le checkpoint vidéo
    weights = {
        key.removeprefix(prefix): value
        for key, value in checkpoint["state_dict"].items()
        if key.startswith(prefix)
    }
    if not weights:
        raise ValueError(f"Aucun poids avec le préfixe {prefix!r}")
    decoder.load_state_dict(weights, strict=True)
    decoder = decoder.to(device).eval()
    decoder.requires_grad_(False)
    return decoder


@torch.no_grad()
def decode_slot_masks(decoder, slots, output_size):
    """[F, S, D] -> masques booléens [F, S, H, W].

    Ici F est traité comme un batch : le décodeur agit image par image.
    Le redimensionnement direct convient aux slots extraits avec un Resize
    carré sans recadrage, comme dans extract_videosaur.py. Si l'extraction
    change, il faut reproduire exactement sa géométrie.
    """
    device = next(decoder.parameters()).device
    x = torch.as_tensor(slots, dtype=torch.float32, device=device)
    masks = decoder(x)["masks"]  # [F, S, P], avec P=196 ici
    n_frames, n_slots, n_patches = masks.shape
    side = int(n_patches ** 0.5)
    if side * side != n_patches:
        raise ValueError("La grille du décodeur doit être carrée.")
    masks = masks.reshape(n_frames, n_slots, side, side)
    masks = F.interpolate(masks, size=output_size, mode="bilinear", align_corners=False)
    labels = masks.argmax(dim=1)
    hard_masks = torch.stack([labels == k for k in range(n_slots)], dim=1)
    return hard_masks.cpu().numpy()


def load_object_masks(proposal_path, annotation_path, frame_index):
    """Retourne masques [O,H,W], métadonnées et taille (H,W).

    Les propositions n'ont pas d'object_id. On les relie aux annotations par
    (couleur, matériau, forme), seulement si l'association est unique.
    Cela reste une proposition d'identité à vérifier visuellement.
    """
    with Path(proposal_path).open() as f:
        proposals = json.load(f)
    with Path(annotation_path).open() as f:
        annotation = json.load(f)
    if proposals["video_index"] != annotation["scene_index"]:
        raise ValueError("Les propositions et annotations concernent deux vidéos différentes.")
    frame = next(f for f in proposals["frames"] if f["frame_index"] == frame_index)
    return decode_object_masks(frame, annotation)


def decode_object_masks(frame, annotation):
    """Même résultat que load_object_masks(), à partir de JSON déjà chargés.

    Utile sur le cluster : chaque JSON est lu une seule fois par vidéo.
    """
    from pycocotools import mask as mask_utils

    objects = frame["objects"]
    if not objects:
        raise ValueError(f"Aucune proposition dans l'image {frame['frame_index']}.")

    attributes = ("color", "material", "shape")
    signatures = [tuple(obj[key] for key in attributes) for obj in objects]
    counts = Counter(signatures)
    masks, metadata = [], []
    for index, obj in enumerate(objects):
        rle = obj["mask"].copy()
        if isinstance(rle["counts"], str):
            rle["counts"] = rle["counts"].encode("utf-8")
        masks.append(mask_utils.decode(rle).astype(bool))
        candidates = [
            gt["object_id"] for gt in annotation["object_property"]
            if tuple(gt[key] for key in attributes) == signatures[index]
        ]
        object_id = candidates[0] if len(candidates) == 1 and counts[signatures[index]] == 1 else None
        metadata.append({
            "proposal_index": index,
            "object_id": object_id,  # None = inconnu ou ambigu, à exclure du matching final
            "color": obj["color"], "material": obj["material"], "shape": obj["shape"],
            "score": obj["score"],
        })
    masks = np.stack(masks)
    return masks, metadata, tuple(masks.shape[-2:])


def compute_iou_matrix(slot_masks, object_masks):
    """Masques [S,H,W] et [O,H,W] -> matrice [S,O], valeurs entre 0 et 1."""
    slot_masks = np.asarray(slot_masks, dtype=bool)
    object_masks = np.asarray(object_masks, dtype=bool)
    if slot_masks.shape[-2:] != object_masks.shape[-2:]:
        raise ValueError("Les deux ensembles doivent avoir la même résolution.")
    iou = np.zeros((len(slot_masks), len(object_masks)))
    for k, slot_mask in enumerate(slot_masks):
        for o, object_mask in enumerate(object_masks):
            intersection = (slot_mask & object_mask).sum()
            union = (slot_mask | object_mask).sum()
            iou[k, o] = intersection / union if union else 0.0
    return iou


def compute_coverage_matrix(slot_masks, object_masks):
    """Retourne coverage[k, o] = fraction de l'objet o couverte par le slot k.

    On divise par la surface de l'OBJET, pas par la surface du slot.
    Exemple : 1.0 signifie que le slot contient tout l'objet, même s'il déborde
    beaucoup sur le fond. Un masque d'objet vide reçoit une couverture de 0.
    """
    slot_masks = np.asarray(slot_masks, dtype=bool)
    object_masks = np.asarray(object_masks, dtype=bool)
    if slot_masks.shape[-2:] != object_masks.shape[-2:]:
        raise ValueError("Les deux ensembles doivent avoir la même résolution.")
    coverage = np.zeros((len(slot_masks), len(object_masks)))
    for k, slot_mask in enumerate(slot_masks):
        for o, object_mask in enumerate(object_masks):
            object_area = object_mask.sum()
            intersection = (slot_mask & object_mask).sum()
            coverage[k, o] = intersection / object_area if object_area else 0.0
    return coverage


def get_pair_coverage(coverage_matrix, slot_index, proposal_index):
    """Retourne (couverture de l'objet associé, couverture maximale d'un autre).

    Tous les autres masques comptent, même si leur object_id est inconnu.
    Avec un seul objet proposé, la couverture des autres vaut 0.
    """
    target_coverage = float(coverage_matrix[slot_index, proposal_index])
    other_coverages = np.delete(coverage_matrix[slot_index], proposal_index)
    other_coverage = float(other_coverages.max()) if other_coverages.size else 0.0
    return target_coverage, other_coverage


def prepare_frame(decoder, video_slots, frame_index, proposal_path, annotation_path):
    """Prépare les masques et les deux matrices pour UNE image.

    video_slots doit appartenir à la même vidéo que les deux JSON. Chaque
    indice temporel doit correspondre à la frame brute (slots extraits sans saut).
    """
    if not 0 <= frame_index < len(video_slots):
        raise ValueError("Indice d'image hors de la vidéo.")
    objects_masks, objects, size = load_object_masks(proposal_path, annotation_path, frame_index)
    slots_masks = decode_slot_masks(decoder, video_slots[frame_index:frame_index + 1], size)[0]
    return {
        "frame_index": frame_index,
        "slot_masks": slots_masks,
        "object_masks": objects_masks,
        "objects": objects,
        "iou": compute_iou_matrix(slots_masks, objects_masks),
        "coverage": compute_coverage_matrix(slots_masks, objects_masks),
    }


def assign_slots(iou_matrix):
    """Retourne [(slot_index, proposal_index), ...] en maximisant l'IoU.

    L'affectation doit être un à un. Elle ne garantit pas de bonnes IoU :
    il faudra ensuite filtrer les couples avec is_reliable().
    """

    slot_index, proposal_index = linear_sum_assignment(iou_matrix, maximize=True)
    return list(zip(slot_index, proposal_index))


def is_reliable(
    coverage_matrix, slot_index, proposal_index,
    min_target_coverage, max_other_coverage,
):
    """Accepte le couple si l'objet est assez couvert et les autres peu couverts.

    Exemple de lecture : min_target_coverage=0.8 et max_other_coverage=0.1
    demandent au moins 80 % de l'objet associé et au plus 10 % de chaque autre.
    Ces valeurs sont à calibrer ; elles ne sont pas imposées par la fonction.
    Vérifier aussi object_id is not None dans les métadonnées de l'appelant.
    """
    target_coverage, other_coverage = get_pair_coverage(
        coverage_matrix, slot_index, proposal_index
    )
    return target_coverage >= min_target_coverage and other_coverage <= max_other_coverage


def build_frame_mapping(result, min_target_coverage, max_other_coverage):
    """Propose les couples par IoU, puis filtre leur séparation spatiale."""
    mapping = {}

    for slot_index, proposal_index in assign_slots(result["iou"]):
        object_id = result["objects"][proposal_index]["object_id"]

        if object_id is None:
            continue

        if is_reliable(
            result["coverage"], slot_index, proposal_index,
            min_target_coverage, max_other_coverage,
        ):
            mapping[int(slot_index)] = object_id

    return mapping


def check_temporal_consistency(reference_mapping, frame_mappings):
    """Vérifie les identités sur une fenêtre.

    reference_mapping : dict {slot_index: object_id}, défini sur l'historique.
    frame_mappings : liste de tels dictionnaires, après filtrage des couples.

    Version simple : retourner True seulement si chaque couple de référence
    est présent et inchangé dans chaque image. Une absence doit faire échouer
    ce contrôle conservateur. Ne pas réordonner les slots pour cacher un switch.
    Exclure les object_id inconnus avant de construire ces dictionnaires.
    """
    if not reference_mapping or not frame_mappings:
        return False
    # Les autres couples peuvent changer ; ceux de référence doivent rester.
    for mapping in frame_mappings:
        for slot_index, object_id in reference_mapping.items():
            if mapping.get(slot_index) != object_id:
                return False

    return True


def show_pair(result, slot_index, proposal_index):
    """Affiche aussi le débordement du slot sur les autres objets proposés."""
    import matplotlib.pyplot as plt

    slot_mask = result["slot_masks"][slot_index]
    object_mask = result["object_masks"][proposal_index]
    obj = result["objects"][proposal_index]
    target_coverage, other_coverage = get_pair_coverage(
        result["coverage"], slot_index, proposal_index
    )
    other_objects = np.zeros_like(object_mask)
    for index, mask in enumerate(result["object_masks"]):
        if index != proposal_index:
            other_objects |= mask
    title = f'{obj["color"]} / {obj["material"]} / {obj["shape"]} (id={obj["object_id"]})'
    fig, axes = plt.subplots(1, 4, figsize=(14, 3))
    titles = [f"Slot {slot_index}", title,
              f"Objet associé couvert : {target_coverage:.1%}",
              f"Autres objets ; max couvert : {other_coverage:.1%}"]
    images = [slot_mask, object_mask, slot_mask & object_mask, slot_mask & other_objects]
    for ax, image, label in zip(axes, images, titles):
        ax.imshow(image, cmap="gray", vmin=0, vmax=1)
        ax.set_title(label, fontsize=9)
        ax.axis("off")
    fig.suptitle(f'Image {result["frame_index"]} ; IoU du couple = {result["iou"][slot_index, proposal_index]:.3f}')
    fig.tight_layout()
    return fig
