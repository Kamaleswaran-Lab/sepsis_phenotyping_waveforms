# sepsis_phenotyping_waveforms

Code for **"Deep learning of continuous ICU waveforms reveals cardiovascular–autonomic sepsis phenotypes with differential outcomes"** (manuscript under review).

The pipeline derives sepsis phenotypes directly from continuous bedside waveforms (ECG, photoplethysmogram, and respiratory impedance) recorded in the 5 minutes before Sepsis-3 onset. It extracts physiomarkers with the Multimodal Physiomarker Toolbox (MPT), learns a representation with an FT-Transformer encoder, and identifies phenotypes with consensus clustering. It then characterizes the phenotypes with SHAP and clinician review, relates them to outcomes, and trains an XGBoost classifier for phenotype assignment.

Applied to 2,174 ICU encounters meeting Sepsis-3 criteria (University of Pittsburgh, 2016–2022), the pipeline identified four physio-phenotypes (SP-1 to SP-4). These phenotypes differed in in-hospital mortality, septic shock, vasopressor use, and mechanical ventilation despite similar baseline severity.

---

## Pipeline overview

```
Stage 0  Sepsis-3 encounters ──► waveform record matching (WFDB)
Stage 1  Up to 72 × 5-min segments over the 6 h before onset ──► MPT physiomarkers per segment
Stage 2  Segment 0 (the 5 min ending at onset) ──► preprocessing ──► FT-Transformer ──► UMAP
         ──► consensus clustering (K = 4) ──► XGBoost classifier + SHAP ──► LLM draft summaries
Stage 3  Outcomes, laboratory profiles, survival analysis (Kaplan–Meier, Cox)
```

## Repository structure

| Path | Purpose |
|---|---|
| `Stage0_waveformID_table_generation.ipynb` | Links Sepsis-3 encounters (2016–2022) to their waveform records |
| `Stage1_comprehensiveFeatureTableCreation.ipynb`, `Stage1_batch2.ipynb`, `Stage1_batch3.ipynb` | Extracts physiomarkers from 5-min segments across the 6 h before onset (run in batches) |
| `EventTableGeneration.ipynb` | Builds clinical event tables: vasopressors, fluid boluses, ventilation, CRRT, ICU stays |
| `Stage2A_phenotype_Identification_for_segments_FTT.ipynb` | **Primary analysis**: preprocessing, imputation, outcomes, FT-Transformer phenotyping |
| `Stage2A_phenotype_Identification_for_segments_VAE.ipynb` | Same pipeline with the deep variational autoencoder representation |
| `Stage2A_phenotype_Identification_for_segments.ipynb` | Comparator representations plus phenotype characterization, labs, and survival analysis |
| `Feature_Distributions_plot.ipynb` | Feature distribution figures |
| `MPT.py`, `MPT_featureExtraction_utilities.py` | Multimodal Physiomarker Toolbox: fiducial detection, signal quality, feature extraction |
| `MPT_physiomarkers_wrapper.py` | Command-line wrapper for MPT feature extraction |
| `MPT_plot_multiwav.py` | Multi-waveform plotting (matplotlib / Plotly) |
| `tc_tabular_consensus_clustering_pipeline_v4.py` | Preprocessing, FT-Transformer, DVAE, UMAP/PCA, consensus clustering, XGBoost + SHAP |
| `tc_consensus_clust.py` | Consensus clustering wrapper and final label assignment |
| `tc_Missingness_Imputation.py`, `tc_MissingnessReport.py` | Imputation utilities and missingness report |
| `integrated_sepsis_pipeline.py`, `sepsis_analysis_utils.py` | End-to-end workflow: clustering, prediction, reports, publication outputs |
| `sepsis_phenotype_interpreter.py` | LLM-assisted draft phenotype interpretation (OpenAI or Anthropic API) |
| `mapping.csv` | Maps original feature names to modality-prefixed IDs (`ECG_`, `HRV_`, `PPG_`, `RESP_`) |
| `sepsis_phenotype_study_results/` | Saved outputs per representation: `FTT_umap_latest` (primary), `VAE_umap_latest`, `PCA_umap_latest`, `All_umap_latest` |
| `Figures/` | Figures (UMAP, Kaplan–Meier, feature correlations) |
| `UNUSED_YET/` | Exploratory analyses not used in the manuscript |

## Method details (as implemented)

**Cohort and window.** Each ICU encounter meets Sepsis-3 criteria and has ECG, PPG, and respiration waveforms. The analysis uses segment 0, the usable 5-minute segment ending at Sepsis-3 onset.

**Features.** MPT extracts beat-level ECG, PPG, and respiratory features. Each is summarized over the segment by mean, 25th/75th percentiles, skewness, and Shannon entropy. Heart-rate-variability indices are single values computed from the segment's RR-interval series, so they are not summarized further. The charted qSOFA indicators (SBP ≤ 100 mmHg, RR ≥ 22/min, GCS < 15) and their total are added. This gives 429 candidate features.

**Preprocessing.** Preprocessing runs in this order:

1. Values beyond 1.5 × IQR are set to missing.
2. Features with ≥ 15% missingness are removed, as are segments with > 80% missingness.
3. Remaining gaps are filled forward and then backward across each encounter's 5-minute segments, and anything still missing gets the column median.
4. Features are standardized, then filtered by variance (threshold 0.01) and pairwise correlation (|r| > 0.85).

This leaves 272 features: 268 waveform-derived plus 4 qSOFA.

**Representation.** An FT-Transformer autoencoder is trained with these settings:

| Setting | Value |
|---|---|
| d_model | 128 |
| Attention heads | 4 |
| Layers | 3 |
| MLP width | 256 |
| CLS projection | 32-d |
| Dropout | 0.1 |
| Epochs | 120 |
| Batch size | 32 |
| Learning rate | 2e-3 |
| Early stopping patience | 20 |
| random_state | 42 |

The 32-d embedding is projected to 2-D with UMAP (spectral initialization, n_neighbors = 30, random_state = 10). The comparators are standardized features + UMAP, PCA (90% variance) + UMAP, and a 32-d DVAE + UMAP.

**Clustering.** Consensus clustering uses `consensusclustering.ConsensusClustering` with agglomerative clustering (Euclidean distance, average linkage). It evaluates K = 2–9 with 100 resamples at 80% subsampling. Final labels come from average-linkage clustering of 1 − consensus matrix, with K = 4.

**Classifier.** An XGBoost classifier (`multi:softprob`) is trained on the same 272 features with an 80/20 split (random_state = 42). SHAP TreeExplainer provides feature importance.

**Interpretation.** GPT-4o produces zero-shot narrative drafts from SHAP-ranked features. These are exploratory. Final phenotype characterizations come from the quantitative analysis of physiomarkers, laboratory values, and outcomes, reviewed by critical-care clinicians.

## Installation

```bash
git clone https://github.com/Kamaleswaran-Lab/sepsis_phenotyping_waveforms.git
cd sepsis_phenotyping_waveforms
pip install -r requirements.txt
pip install tensorflow==2.13 umap-learn==0.5.5 scikit-learn==1.3.2 scipy==1.11.4 \
            consensusclustering xgboost shap scikit-plot lifelines seaborn missingno tqdm
```

`requirements.txt` covers the waveform-processing dependencies. The second line adds the modelling stack used in the notebooks. `scipy` is pinned because `scikit-plot` needs `scipy.interp`, which the pipeline module also patches.

## Usage

### 1. Physiomarker extraction for one recording

```bash
python MPT_physiomarkers_wrapper.py --ecg ecg.npy --ppg ppg.npy --resp resp.npy --fs 125 --stat_method all
```

Inputs are NumPy arrays for a single segment; omit a modality to skip it. Outputs are written to `./Output_Data/`. The `--sample` option expects a BIDMC record at `./Sample_Data/bidmc09m`, which is not included and can be downloaded from [PhysioNet](https://physionet.org/content/bidmc/1.0.0/).

### 2. Full study pipeline

Run the notebooks in order: Stage 0, Stage 1 (all batches), `EventTableGeneration`, then `Stage2A_..._FTT` for the primary analysis. The notebooks expect this layout:

```
parent/
├── sepsis_phenotyping_waveforms/      # this repository (notebook working directory)
│   └── Data_folder/                   # step-1/step-2 feature tables and Event_Table/
├── pitts_emr_data/                    # EMR extracts: vitals, labs, units, ventilation
└── Sepsis_PittsSuper_table_2_12_24.csv
```

### 3. Phenotyping on your own feature table

```python
from sepsis_analysis_utils import run_complete_sepsis_analysis_workflow

results, report, table, cards = run_complete_sepsis_analysis_workflow(
    df_data=df_features,                 # one row per encounter
    feature_columns=feature_cols,
    standard_minmax="standard", var_thresh=0.01, corr_thresh=0.85,
    dim_reduce_method="custom_embedding", embedding_type="ftt_umap", k_opt=4,
    api_provider="openai", api_key=os.environ["OPENAI_API_KEY"], model_name="gpt-4o",
    output_base_dir="./sepsis_phenotype_study_results/my_run",
)
```

Other options for `embedding_type` are `vae_umap` and `ftt`. For the non-deep methods, set `dim_reduce_method` to `umap` or `pca_umap` instead. The LLM step needs an API key; read it from an environment variable and never commit it.

## Outputs

Each folder in `sepsis_phenotype_study_results/` contains:

- **`analysis_results/`** holds the core outputs:
  - `cluster_assignments.csv`: one label per encounter, in row order of the input
  - `selected_features.csv`: the 272 features that entered clustering
  - `consensus_clust.json`: classifier metrics and SHAP values
  - `phenotype_interpretations.json`: GPT-4o drafts
  - `analysis_summary.json`
- **`publication_materials/`** holds the methods text, phenotype summary table, and clinical cards.

In `FTT_umap_latest`, clusters 0, 1, 2 and 3 correspond to SP-1, SP-2, SP-3 and SP-4 in the manuscript. Their sizes are 580, 721, 601 and 272.

## Reproducibility notes

- The resampling step of consensus clustering was not given a random seed in the published run, so labels can differ slightly between runs. TensorFlow training may also vary on GPU.
- To reproduce within this variability, fix the FT-Transformer and UMAP seeds as above and pass `rng` to `ConsensusClustering`.
- Summarize agreement between runs with the adjusted Rand index.

## Data availability

The clinical and waveform data come from a retrospectively collected ICU cohort at the University of Pittsburgh (2016–2022). They are not publicly available because of patient-privacy and institutional data-governance restrictions. Access requires institutional approval, a data use agreement, and a reasonable request to the corresponding authors. This repository contains code and aggregate outputs only; no patient-level data are included.

## Citation

If you use this code, please cite the manuscript (citation details will be added on publication):

.........

## License

MIT License. See [LICENSE](LICENSE).

## Contact

Tilendra Choudhary (tilendra.choudhary@duke.edu) and Rishikesan Kamaleswaran (r.kamaleswaran@duke.edu), Duke University School of Medicine.
