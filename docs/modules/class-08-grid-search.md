# Class 08 — Hyperparameter Search with GridSearchCV

## Source status

This module is a curated consolidation of the Class 08 README extracted by the file-reading tool. Two uploaded files shared the name `5-README.md`; only the 31,185-byte Class 09 version was available under that filename in the execution environment. Consequently, this document is not a byte-for-byte copy of the 35,138-byte Class 08 original. The original notebook, CSV, and PDF are not included.

## Laboratory described by the supplied README

- Dataset: `5_musicas.csv`, binary target `hit`.
- Remove `id`, `titulo`, `artista`, and `popularidade`; the source identifies `popularidade` as leakage.
- Source workflow: one-hot encode `genero` with `pd.get_dummies`, then create a stratified 70/30 split with seed 0.
- Pipeline: `StandardScaler` → `LogisticRegression(max_iter=2000)`.
- Grid: `clf__C=[0.01,0.1,1,10,100]`, `clf__class_weight=[None,"balanced"]`.
- Validation: five-fold `StratifiedKFold(shuffle=True, random_state=0)`.
- Primary selection metric: Average Precision.

## Results reported by the supplied README, not rerun here

| Item | Source-reported value |
|---|---:|
| Prepared matrix | 1,050 rows × 18 features |
| Hit prevalence | 0.122 |
| Train / test rows | 735 / 315 |
| Test positives | 38 |
| Candidate configurations | 10 |
| CV fits, excluding final refit | 50 |
| Best internal Average Precision | 0.7207 |
| Final-test Average Precision | 0.7116 |
| Final-test ROC AUC | 0.9296 |
| Candidates within champion's one-SD band | 10 of 10 |

Winning parameters: `clf__C=1`, `clf__class_weight=None`.

## Interpretation and limitations

Internal best score selects; the protected test confirms. All candidates in a one-SD band indicate a broad measured plateau, not a formal significance test. The scores are specific to this experiment and cannot be compared directly with Class 06 or Class 07 accuracy.

The source's encoding-before-split workflow is documented faithfully. A recommended improvement is to fit categorical preprocessing inside a `ColumnTransformer` and `Pipeline`, treating unseen categories explicitly. This improvement is not claimed to have been implemented in the classroom workflow.

## Presentation

https://lively-halva-00f14e.netlify.app/
