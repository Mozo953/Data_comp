import os
import shutil
from pathlib import Path

base = Path(".")

# ============ artifacts_extratrees_corr_optuna ============
corr_dir = base / "artifacts_extratrees_corr_optuna"

# Créer tous les dossiers
(corr_dir / "03_archive_old_tests").mkdir(exist_ok=True, parents=True)
(corr_dir / "04_other").mkdir(exist_ok=True, parents=True)
(corr_dir / "01_experiments_SAFE" / "q25_feat36_corr988_cv6_trials20").mkdir(exist_ok=True, parents=True)
(corr_dir / "02_experiments_OPEN" / "q20_feat45_corr990_cv6_trials24").mkdir(exist_ok=True, parents=True)
(corr_dir / "02_experiments_OPEN" / "q20_feat45_corr990_cv10_strongcheck").mkdir(exist_ok=True, parents=True)
(corr_dir / "02_experiments_OPEN" / "q20_feat45_corr990_cv10_seeds").mkdir(exist_ok=True, parents=True)

count = 0

# Déplacer fichiers SAFE
for f in list(corr_dir.glob("optuna_explicite_SAFE_*")):
    if f.is_file():
        dest = corr_dir / "01_experiments_SAFE" / "q25_feat36_corr988_cv6_trials20" / f.name
        shutil.move(str(f), str(dest))
        count += 1
        print(f"✓ {f.name}")

# Déplacer fichiers OPEN - cv6_trials24
for f in list(corr_dir.glob("optuna_explicite_OPEN_q20_feat45_corr990_cv6_trials24_*")):
    if f.is_file():
        dest = corr_dir / "02_experiments_OPEN" / "q20_feat45_corr990_cv6_trials24" / f.name
        shutil.move(str(f), str(dest))
        count += 1
        print(f"✓ {f.name}")

# Déplacer fichiers OPEN - cv10 strongcheck
for f in list(corr_dir.glob("optuna_explicite_OPEN_q20_feat45_corr990_cv10_strongcheck_*")):
    if f.is_file():
        dest = corr_dir / "02_experiments_OPEN" / "q20_feat45_corr990_cv10_strongcheck" / f.name
        shutil.move(str(f), str(dest))
        count += 1
        print(f"✓ {f.name}")

# Déplacer fichiers OPEN - cv10 seeds
for f in list(corr_dir.glob("optuna_explicite_OPEN_*_seed*")):
    if f.is_file() and "cv10" in f.name:
        dest = corr_dir / "02_experiments_OPEN" / "q20_feat45_corr990_cv10_seeds" / f.name
        shutil.move(str(f), str(dest))
        count += 1
        print(f"✓ {f.name}")

# Déplacer fichiers archive
for f in list(corr_dir.glob("extratrees_*")):
    if f.is_file():
        dest = corr_dir / "03_archive_old_tests" / f.name
        shutil.move(str(f), str(dest))
        count += 1
        print(f"✓ {f.name}")

# Déplacer fichiers other
for fname in ["optuna_objectif_0.04_EX_Type_melchior_20260407T193613Z.json", "A TESTER DEMAIN.csv"]:
    src = corr_dir / fname
    if src.exists():
        dest = corr_dir / "04_other" / fname
        shutil.move(str(src), str(dest))
        count += 1
        print(f"✓ {fname}")

# ============ artifacts_extratrees_featurebomb ============
fb_dir = base / "artifacts_extratrees_featurebomb"

# Créer les dossiers
(fb_dir / "01_auto_experiments").mkdir(exist_ok=True, parents=True)
(fb_dir / "02_featurebomb_experiments").mkdir(exist_ok=True, parents=True)

# Déplacer fichiers auto
for f in list(fb_dir.glob("auto_*")):
    if f.is_file():
        dest = fb_dir / "01_auto_experiments" / f.name
        shutil.move(str(f), str(dest))
        count += 1
        print(f"✓ {f.name}")

# Déplacer fichiers featurebomb
for f in list(fb_dir.glob("extratrees_featurebomb_*")):
    if f.is_file():
        dest = fb_dir / "02_featurebomb_experiments" / f.name
        shutil.move(str(f), str(dest))
        count += 1
        print(f"✓ {f.name}")

print(f"\n✅ {count} fichiers organisés!")
