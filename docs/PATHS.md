# Chemins pour Fir

Tous les chemins sont dans **`configs/paths.fir.yaml`**. Remplacer
`TON_IDENTIFIANT` par ton compte Fir et adapter les emplacements reels des
poids, slots, videos, annotations, propositions et resultats.
Ce fichier est versionne pour suivre le code lors des `git pull` sur Fir.

Les noms proposes correspondent a VideoSAUR sur CLEVRER. Garder les slots,
l'encodeur et les predicteurs compatibles. Les parametres scientifiques
(historique, horizon, pas temporel, seuils, seed) doivent etre versionnes dans
la configuration de l'experience, separement de ces chemins.

`probing/clevrer/calibrate_matching.py` lit ce fichier avec `--paths`.
Les scripts d'origine du depot continuent d'utiliser leurs propres arguments
et configurations Hydra.

`object_masks_dir` doit pointer vers le dossier
contenant les `proposal_XXXXX.json` de `derender_proposals`. La recherche des
annotations et propositions est recursive : les sous-dossiers sont acceptes.

Pour telecharger les donnees du projet dans les dossiers configures :

```bash
bash scripts/ablation_robotic_MVA/download_data_fir.sh --dry-run
bash scripts/ablation_robotic_MVA/download_data_fir.sh
```

Le script recupere les slots VideoSAUR CLEVRER, les trois checkpoints, les
annotations et les videos de validation, ainsi que les propositions de masques.
Il reprend les telechargements interrompus dans des fichiers `.part`. Il saute
les fichiers deja presents et les dossiers contenant les 5 000 identifiants
de validation. Apres extraction, un marqueur memorise le nombre de fichiers
effectivement presents dans l'archive officielle ; une relance ne les extrait
pas de nouveau. Les fichiers absents de la source restent signales par le
pretraitement. `--skip-videos` permet de ne pas telecharger les MP4.
Les archives sont conservees dans `downloads/`, a cote du fichier de slots.
Le script necessite Python avec OmegaConf, curl et unzip, et un acces Internet.

Si les videos sont deja dezippees sous `clevrer_val/video_10000-11000/`, etc.,
mettre `videos_dir` au dossier parent `clevrer_val`. Ne pas deplacer les MP4.
Le pretraitement utilise uniquement les slots et les JSON, pas les MP4. Pour
les controles visuels, `match_objets.find_video_path(videos_dir, 10000)` retrouve
le fichier dans les sous-dossiers.

Depuis la racine du depot, apres activation de l'environnement Python et des
modules necessaires sur Fir :

```bash
sbatch --account=TON_COMPTE scripts/ablation_robotic_MVA/preprocess_validation_fir.sbatch
```

Ajouter `--max-videos 5` a cette commande pour un petit essai. Sans cette
option, le script parcourt toute la validation, toutes ses collisions, sans
tirage aleatoire. Une nouvelle soumission avec les memes chemins et parametres
reprend les videos deja terminees. Ne pas lancer deux jobs sur le meme dossier
de sortie en parallele. Si les donnees ou la methode changent, choisir un
nouveau dossier de sortie.

Les resultats sont dans `paths.output_dir` : `videos/` contient le cache par
video et les correspondances objet-slot par fenetre ; `windows.jsonl` regroupe
les mesures ; `exclusions.csv` conserve les motifs d'exclusion ; `thresholds.csv`
contient les nombres et taux de conservation. La matrice est affichee dans le
log Slurm et sauvegardee dans `threshold_matrix.csv` et `threshold_matrix.png`.
`summary.json` indique aussi les videos manquantes ou encore non calculees.

Pour modifier uniquement la grille de seuils dans `configs/calibrate_matching.yaml`
et reconstruire les tableaux sans GPU, sans recharger les gros fichiers :

```bash
python -m probing.clevrer.calibrate_matching --summary-only
```

Le calcul sur toute la validation ne choisit pas automatiquement les seuils.
Reserver des videos au developpement pour les choisir, puis evaluer l'ablation
sur des videos distinctes avec les seuils figes.

Le code se synchronise via Git ; les poids et donnees se telechargent ou se
transferent separement. Les resultats importants doivent aussi etre conserves
sur un espace durable : scratch sert de stockage de travail temporaire.
