# Ablation de collision : mode d'emploi

## L'idée en une phrase
Pour une collision entre A et B, on cache l'historique d'**un seul autre objet à la fois** et on regarde de combien la prédiction de la cible (A ou B) se dégrade.
Si le modèle se sert de l'interaction, cacher le **partenaire** doit faire plus mal que cacher n'importe quel autre objet.

## Comment on procède
1. **Fenêtres** : `matching_preprocess_val/windows.jsonl`, une collision par ligne, avec l'appariement slot ↔ objet.
   - Le modèle voit les frames T−11, T−9, …, T−1 (6 pas) et prédit T+1, T+3, … (10 pas).
2. **Ablation** (`run_ablation.py`) : pour chaque fenêtre, on fait tourner le prédicteur gelé plusieurs fois, en une seule passe batchée.
   - **Référence** : tout l'historique est visible.
   - **Slot k caché** : l'historique de k est remplacé par des tokens masqués aux pas 1–5 (l'ancre t=0 reste visible).
   - **Variante « cible cachée »** : on cache la cible **et** k en même temps.
   - Erreur = ‖ẑ − z‖² dans l'espace des slots.
3. **Analyse** (`analyze.py`) : on garde les fenêtres bien appariées, puis on crée deux instances par fenêtre (cible = A, puis cible = B).
   - Pour chaque instance : `D(k) = err(cible | k caché) − err(cible | référence)`.
   - **Succès** si le partenaire a le plus grand D parmi les candidats.

## Lancer (dans l'ordre)
```bash
# 1. Extraire les slots utiles du gros pkl (une seule fois, lent : charge tout le fichier)
uv run python -m ablation.extract_slots

# 2. Test sur une fenêtre, avec vérifications, sans toucher à results/
uv run python -m ablation.run_ablation --model cjepa \
    --windows matching_preprocess_val/windows_example.json --results-dir /tmp/abl_test --check

# 3. Run complet, pour les deux modèles
#    (ajouter --target-hidden pour la variante « cible cachée »)
uv run python -m ablation.run_ablation --model cjepa
uv run python -m ablation.run_ablation --model ocjepa

# 4. Tableaux
uv run python -m ablation.analyze
```
Options de `run_ablation` :
- `--device {cpu,mps}` : `cpu` suffit ;
- `--limit N` : seulement les N premières fenêtres ;
- `--horizons 1 3 5` : horizons sauvegardés (par défaut 1 à 10). h = 1 est toujours calculé pour le résultat principal ;
- `--target-hidden` : ajoute la variante « cible cachée ». Elle demande 49 masques de plus, donc environ 7 fois plus de temps. Désactivée par défaut.

Options de `analyze` :
- `--horizons 1 3` : horizons à mettre dans `table_horizon.csv`. Par défaut, tous ceux qui ont été sauvegardés.
- L'agrégat `sum_1-5` n'est calculé que si les horizons 1 à 5 sont tous présents.
- La variante « cible cachée » n'est analysée que si le run l'a calculée.

Si tu relances avec une autre liste de fenêtres, supprime d'abord `results/errors_index.csv` : le script refuse de le mélanger avec une autre liste.

## Ce qui sort (`results/`)
| Fichier | Contenu |
|---|---|
| `errors_<model>.npz` | Erreurs brutes : `err` [n, 8, 7] (h = 1), `horizons`, `err_h` (horizons choisis), `err_copy` (baseline « copie du dernier slot »), `err_tgt_hidden` (seulement avec `--target-hidden`) |
| `errors_index.csv` | Ligne i ↔ fenêtre i (`video_id`, `event_index`, …) |
| `table_main.csv` | **Résultat principal** : seuil 95/5, h = 1 |
| `table_robustness.csv` | Idem au seuil 90/10 |
| `table_by_candidates.csv` | Top-1 selon le nombre de candidats |
| `table_horizon.csv` | Exploratoire : h = 1…10, somme sur h = 1…5, standard et cible cachée |
| `instances.csv` | Une ligne par instance, pour les figures |

Colonnes principales :
- `top1` : taux de succès, avec son IC 95 % (`_ci_lo`, `_ci_hi`) ;
- `chance` : le hasard, soit 1 / nombre de candidats ;
- `frac_d_partner_pos` : part des instances où cacher le partenaire dégrade la prédiction ;
- `diff_partner_others_*` : D(partenaire) − D(autres) ;
- `ratio_partner_self_*` : D(partenaire) / D(cible cachée).

## Règles figées (voir `CLAUDE.md`, Journal)
- **Fenêtre acceptée** : `status == "measured"`, les trois flags de qualité vrais, `min_target_coverage ≥ 0.95` et `max_other_coverage ≤ 0.05` (robustesse : 0.90 / 0.10).
- **Split** : seules les vidéos 10100–14999 (eval) comptent ; 10000–10099 (dev) sert au debug.
- **Candidats** : les objets visibles sur toute la fenêtre, sauf la cible. Les slots de fond ne sont jamais candidats.
- **IC 95 %** : bootstrap sur les **vidéos**, pas sur les instances.
- **Résultat principal = h = 1.** Les autres horizons et la variante cible cachée sont **exploratoires**.
- **Jamais de sélection de fenêtres d'après l'erreur du modèle.**

## Fichiers
- `config.py` : tous les chemins et constantes. C'est la seule source de vérité.
- `windows.py` : lecture de `windows.jsonl`.
- `extract_slots.py` : extraction des slots depuis le gros pkl.
- `run_ablation.py` : passes du modèle et calcul des erreurs.
- `analyze.py` : filtres, instances, métriques et tableaux.

## Limites à garder en tête
- OC-JEPA n'a jamais vu de masques à l'entraînement : pour lui, cacher un slot est hors distribution.
- L'identité slot ↔ objet n'est vérifiée que jusqu'à T+1. Pour h > 1, c'est une hypothèse.
- À h = 1, les modèles font moins bien que la baseline « copie du dernier slot » : l'effet de la collision y est encore invisible.
