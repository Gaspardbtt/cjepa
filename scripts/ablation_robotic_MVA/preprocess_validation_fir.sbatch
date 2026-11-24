#!/bin/bash
#SBATCH --job-name=cjepa-matching-val
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-preprocess-%j.out

# Soumettre DEPUIS LA RACINE DU DEPOT, avec ton compte d'allocation :
# sbatch --account=TON_COMPTE scripts/ablation_robotic_MVA/preprocess_validation_fir.sbatch
#
# Avant sbatch, charger les modules nécessaires à ton environnement Fir.
# Activer ton environnement Python, ou exporter le chemin de son venv :
# export CJEPA_ENV=/chemin/vers/ton/venv
# Les ressources ci-dessus sont un point de départ à ajuster après un petit essai.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Soumettre depuis la racine de cjepa}"
if [[ ! -f configs/calibrate_matching.yaml ]]; then
    echo "Le répertoire de soumission doit être la racine de cjepa." >&2
    exit 1
fi
if [[ -n "${CJEPA_ENV:-}" ]]; then
    source "${CJEPA_ENV}/bin/activate"
fi
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export MPLCONFIGDIR="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/cjepa-matplotlib"
mkdir -p "$MPLCONFIGDIR"

# Pas de --max-videos : le programme traite toute la validation (sans seed).
# Les arguments optionnels sont transmis, par exemple --max-videos 5.
srun python -m probing.clevrer.calibrate_matching \
    --config configs/calibrate_matching.yaml \
    --paths configs/paths.fir.yaml \
    --device cuda \
    "$@"
