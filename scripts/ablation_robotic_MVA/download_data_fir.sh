#!/usr/bin/env bash
# A exécuter depuis un noeud ayant accès à Internet, sans allocation GPU.
# Usage : bash scripts/ablation_robotic_MVA/download_data_fir.sh
# Options : --paths autre.yaml, --skip-videos, --dry-run.
# Activer d'abord l'environnement Python du projet (OmegaConf doit être installé).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${CJEPA_PYTHON:-python}"
DRY_RUN=false

# Version publiée contenant les slots et les checkpoints utilisés dans ce projet.
HF_BASE="https://huggingface.co/HazelNam/CJEPA/resolve/b1283c683f2591327054088a1413d9a54cb3fdab"
CLEVRER_BASE="https://data.csail.mit.edu/clevrer"


download_file() {
    local url="$1" destination="$2"
    if [[ -s "$destination" ]]; then
        echo "Déjà présent : $destination"
        return
    fi
    echo "Téléchargement : $url"
    echo "             -> $destination"
    if [[ "$DRY_RUN" == true ]]; then
        return
    fi
    mkdir -p "$(dirname "$destination")"
    # Une interruption laisse .part ; une relance reprend le téléchargement.
    curl --fail --location --retry 3 --connect-timeout 30 \
        --continue-at - --output "${destination}.part" "$url"
    mv "${destination}.part" "$destination"
}


count_validation_files() {
    # Compter les identifiants uniques 10000..14999, dans TOUS les sous-dossiers.
    "$PYTHON_BIN" - "$1" "$2" <<'PY'
import sys
from pathlib import Path

root, pattern = Path(sys.argv[1]), sys.argv[2]
ids = set()
for path in root.rglob(pattern):
    video_id = int(path.stem.split('_')[-1])
    if 10000 <= video_id < 15000 and path.is_file() and path.stat().st_size > 0:
        ids.add(video_id)
print(len(ids))
PY
}


download_archive() {
    local url="$1" destination="$2" pattern="$3" expected="$4"
    local count archive marker
    marker="$destination/.${url##*/}.complete"
    if [[ -f "$marker" ]]; then
        # Le nombre fourni par l'archive peut être inférieur à 5000.
        local previous_count
        previous_count="$(cat "$marker")"
        if [[ "$previous_count" =~ ^[0-9]+$ ]] && (( previous_count > 0 )); then
            expected="$previous_count"
        fi
    fi
    count="$(count_validation_files "$destination" "$pattern")"
    if (( count >= expected )); then
        echo "Déjà extrait : $destination ($count fichiers de validation)"
        return
    fi
    archive="$ARCHIVE_DIR/${url##*/}"
    download_file "$url" "$archive"
    echo "Extraction vers : $destination (sous-dossiers conservés)"
    if [[ "$DRY_RUN" == true ]]; then
        return
    fi
    mkdir -p "$destination"
    # Vérifier ce que contient réellement l'archive, sans supposer que tous
    # les masques du dataset ont forcément été publiés.
    expected="$("$PYTHON_BIN" - "$archive" "$pattern" <<'PY'
import fnmatch
import sys
import zipfile
from pathlib import PurePosixPath

ids = set()
with zipfile.ZipFile(sys.argv[1]) as archive:
    for info in archive.infolist():
        name = PurePosixPath(info.filename)
        if fnmatch.fnmatch(name.name, sys.argv[2]) and info.file_size > 0:
            video_id = int(name.stem.split('_')[-1])
            if 10000 <= video_id < 15000:
                ids.add(video_id)
print(len(ids))
PY
    )"
    if (( expected == 0 )); then
        echo "Aucun fichier de validation correspondant à $pattern dans $archive" >&2
        return 1
    fi
    # Réextraction complète si nécessaire, pour remplacer aussi un fichier partiel.
    # Une éventuelle racine supplémentaire dans le ZIP reste compatible avec rglob.
    unzip -oq "$archive" -d "$destination"
    count="$(count_validation_files "$destination" "$pattern")"
    if (( count < expected )); then
        echo "Extraction incomplète : $count/$expected fichiers dans $destination" >&2
        echo "Vérifier l'archive conservée dans $archive" >&2
        return 1
    fi
    printf '%s\n' "$expected" > "$marker"
    echo "Prêt : $destination ($count fichiers de validation)"
}


main() {
    local paths_file="$REPO_ROOT/configs/paths.fir.yaml"
    local skip_videos=false
    while (( $# > 0 )); do
        case "$1" in
            --paths)
                if (( $# < 2 )); then echo "--paths attend un fichier YAML." >&2; return 1; fi
                paths_file="$2"; shift 2 ;;
            --skip-videos) skip_videos=true; shift ;;
            --dry-run) DRY_RUN=true; shift ;;
            -h|--help)
                echo "Usage : bash $0 [--paths fichier.yaml] [--skip-videos] [--dry-run]"
                echo "Les destinations sont lues dans configs/paths.fir.yaml."
                return ;;
            *) echo "Argument inconnu : $1" >&2; return 1 ;;
        esac
    done
    command -v "$PYTHON_BIN" >/dev/null
    if [[ "$DRY_RUN" == false ]]; then
        command -v curl >/dev/null
        command -v unzip >/dev/null
    fi
    cd "$REPO_ROOT"

    # Lire le YAML, sans recopier ses chemins dans ce script.
    local slots encoder cjepa ocjepa videos annotations masks output
    IFS=$'\t' read -r slots encoder cjepa ocjepa videos annotations masks output < <(
        "$PYTHON_BIN" - "$paths_file" "$REPO_ROOT" <<'PY'
import os
import sys
from pathlib import Path
from omegaconf import OmegaConf

config = OmegaConf.load(sys.argv[1]).paths
keys = ['slots', 'encoder_checkpoint', 'cjepa_checkpoint', 'ocjepa_checkpoint',
        'videos_dir', 'annotations_dir', 'object_masks_dir', 'output_dir']
paths = []
for key in keys:
    value = os.path.expandvars(str(config[key]))
    if 'TON_IDENTIFIANT' in value:
        raise ValueError(f'Remplir {key} dans {sys.argv[1]} avant le téléchargement.')
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(sys.argv[2]) / path
    paths.append(str(path.resolve()))
print('\t'.join(paths))
PY
    )
    ARCHIVE_DIR="$(dirname "$slots")/downloads"

    download_file "$HF_BASE/clevrer_videosaur_slots.pkl" "$slots"
    download_file "$HF_BASE/clevrer_videosaur_model.ckpt" "$encoder"
    download_file "$HF_BASE/cjepa-ckpts/clevrer_videosaur_4_epoch_30_object.ckpt" "$cjepa"
    download_file "$HF_BASE/cjepa-ckpts/clevrer_videosaur_0_epoch_30_object.ckpt" "$ocjepa"
    download_archive "$CLEVRER_BASE/annotations/validation/annotation_validation.zip" \
        "$annotations" 'annotation_*.json' 5000
    # L'archive officielle contient aussi des propositions hors validation.
    download_archive "$CLEVRER_BASE/derender_proposals.zip" \
        "$masks" 'proposal_*.json' 5000
    if [[ "$skip_videos" == false ]]; then
        download_archive "$CLEVRER_BASE/videos/validation/video_validation.zip" \
            "$videos" 'video_*.mp4' 5000
    fi

    if [[ "$DRY_RUN" == true ]]; then
        echo "Simulation terminée : aucun téléchargement ni extraction."
    else
        mkdir -p "$output"
        echo "Données prêtes. Les chemins restent ceux de $paths_file."
        echo "Vidéos de validation : $videos (les sous-dossiers sont conservés)."
        echo "Tu peux lancer scripts/ablation_robotic_MVA/preprocess_validation_fir.sbatch."
    fi
}


if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
