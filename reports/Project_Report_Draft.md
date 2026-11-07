# COMPARISON OF SNORT (SIGNATURE-BASED IDS) AND MACHINE LEARNING-BASED INTRUSION DETECTION SYSTEMS ON CICIDS2017

**Final Year Project Report Draft**

---

## ABSTRACT
*(To be written at the end)*

This project aims to compare the performance of the signature-based Intrusion Detection System (IDS), Snort, with machine learning-based approaches (Random Forest, XGBoost and Multilayer Perceptron) on the CICIDS2017 dataset. Flow-based features extracted by CICFlowMeter were used to train and evaluate the ML models. Due to storage and computational constraints, Snort will be evaluated on a single representative PCAP (Friday-WorkingHours traffic) and compared against all three ML models on the exact same subset of flows from the held-out test set.

---

## 1. INTRODUCTION

### 1.1 Background
Intrusion Detection Systems (IDS) are used to monitor network traffic for malicious activities. Signature-based IDS such as Snort rely on predefined rules to detect known attacks, whilst machine learning-based IDS learn patterns from data and can potentially detect previously unseen or variant attacks.

### 1.2 Problem Statement
Most existing studies evaluate signature-based and ML-based IDS separately. There is a need for a fair comparison between Snort and ML models on the same traffic with the same ground truth labels.

### 1.3 Aim
To evaluate and compare the performance of Snort (rule-based) and machine learning models (Random Forest, XGBoost and MLP) for network intrusion detection on CICIDS2017.

### 1.4 Objectives
1. To preprocess and clean the CICIDS2017 flow dataset.
2. To split the dataset into training, validation and test sets using stratified sampling.
3. To train and evaluate Random Forest, XGBoost and MLP classifiers.
4. To apply threshold tuning to optimise the models for intrusion detection.
5. To run Snort on a selected CICIDS2017 PCAP and correlate alerts to flows.
6. To compare Snort against the ML models on the same subset of traffic using standard IDS metrics.

### 1.5 Scope of the Project
- Dataset: CICIDS2017 (Canadian Institute for Cybersecurity)
- ML Approach: Supervised binary classification (Benign vs Attack)
- Models: Random Forest, XGBoost, Multilayer Perceptron (MLP)
- Snort Comparison: Limited to Friday-WorkingHours traffic due to PCAP size constraints. Only flows from the Friday PCAP that appear in the held-out test set will be used for comparison.
- Metrics: Accuracy, Precision, Recall, F1-Score, False Positive Rate (FPR), Average Precision (PR-AUC), Confusion Matrix and per-attack recall.

---

## 2. LITERATURE REVIEW
*(To be completed - Review existing work on CICIDS2017, Snort vs ML comparisons, and flow-based IDS)*

---

## 3. METHODOLOGY

### 3.1 Dataset
The CICIDS2017 dataset contains benign and up-to-date common attacks. The dataset consists of 8 CSV files generated from packet captures using CICFlowMeter. The original dataset share used in this project contained the following files:
- Monday-WorkingHours.pcap_ISCX.csv
- Tuesday-WorkingHours.pcap_ISCX.csv
- Wednesday-workingHours.pcap_ISCX.csv
- Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
- Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
- Friday-WorkingHours-Morning.pcap_ISCX.csv
- Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
- Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv

### 3.2 Data Preprocessing
The following preprocessing steps were carried out in `src/clean_dataset.py`:

1. **Header Cleaning**: Column names were stripped of whitespace.
2. **Label Validation**: Rows with missing or empty labels were rejected.
3. **Duplicate Header Removal**: The duplicate column `Fwd Header Length.1` was verified to be identical to `Fwd Header Length` and dropped.
4. **Numeric Conversion**: Features were coerced to numeric values.
5. **Invalid Row Removal**: Rows containing non-finite values were removed and logged.
6. **Conflicting Label Resolution**: Identical feature vectors with different binary labels were quarantined to `conflicting_rows.pkl` to avoid ambiguity.
7. **Duplicate Removal**: Exact duplicate flow records with the same binary target were removed, keeping the first occurrence based on sorted file order.
8. **Binary Labelling**: A binary target `target = 1` was assigned for any label != `BENIGN`, else `target = 0`. Original attack labels were preserved in `original_label` for per-attack analysis.

From the cleaning report:
- Valid rows before deduplication: 2,827,876
- Invalid rows removed: As logged in `reports/cleaning_report.json`
- Conflicting rows quarantined: 6,666
- Duplicate rows removed: 301,806
- Final rows after cleaning: 2,519,404
- Features used: 77 flow-based features (see `reports/cleaning_report.json`)

### 3.3 Data Splitting
Splitting was performed in `src/create_splits.py` using stratified random sampling based on `original_label` to preserve attack class distributions across splits.

- **Seed**: 42
- **Split Ratio**: 60% Train / 20% Validation / 20% Test
- **Method**: `train_test_split` stratified by original attack label
- **Reproducibility**: Split indices were saved to `data/processed/split_indices.npz` and the SHA256 hash of `cleaned_flows.pkl` was recorded in `reports/split_report.json`. All subsequent scripts verify this hash before loading.

### 3.4 Model Selection and Training

#### 3.4.1 Random Forest
- Algorithm: `sklearn.ensemble.RandomForestClassifier`
- Parameters: `n_estimators=100`, `max_depth=16`, `min_samples_leaf=5`, `max_features='sqrt'`, `class_weight='balanced'`, `n_jobs=2`, `random_state=42`
- Trained on 32-bit float features, evaluated with threshold 0.5 initially

#### 3.4.2 XGBoost
- Algorithm: `xgboost.XGBClassifier`
- Parameters: `tree_method='hist'`, `device='cpu'`, `n_estimators=200`, `max_depth=6`, `learning_rate=0.1`, `min_child_weight=5`, `subsample=0.8`, `colsample_bytree=0.8`, `reg_lambda=1.0`, `scale_pos_weight` = (count_benign/count_attack), `n_jobs=2`, `random_state=42`
- Model saved as `models/xgboost_baseline.ubj` with metadata in `models/xgboost_baseline_metadata.json`

#### 3.4.3 Multilayer Perceptron (MLP)
- Algorithm: `sklearn.neural_network.MLPClassifier`
- Architecture: Hidden layers (32, 16), ReLU activation, Adam solver
- Preprocessing: `StandardScaler` fitted only on training data
- Training: Custom mini-batch training with `partial_fit`, manual epoch shuffling, class-weighted sample weights, early stopping based on validation F1 (patience=4), best model checkpointing
- Batch size: 4096, Max epochs: 20

### 3.5 Threshold Tuning
All models were tuned post-training by sweeping thresholds from 0.3 to 0.7 in steps of 0.05 using validation set probabilities. For each threshold, F1-score was computed and the threshold that maximised validation F1 was selected. The selected thresholds were then evaluated on the locked test set using identical metrics.

- **Random Forest**: Selected threshold = 0.6
- **XGBoost**: Selected threshold = 0.7
- **Neural Network**: Selected threshold = 0.7

### 3.6 Snort vs ML Comparison Plan
*(To be executed once PCAP is available)*

1. **PCAP Selection**: Download and use `Friday-WorkingHours.pcap` (or the full Friday capture containing DDoS traffic). Only flows corresponding to `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv` that exist in the held-out test set will be considered.
2. **Snort Setup**: Run Snort in IDS mode using the Snort Community Ruleset. Record Snort version and ruleset version. Output alerts to `fast.log`/alert files.
3. **Flow Extraction Context**: CICIDS2017 flows were generated by CICFlowMeter from the PCAPs. Correlation will be attempted using the 5-tuple (Source IP, Destination IP, Source Port, Destination Port, Protocol) and timestamp proximity. Where possible, IP/Port/Timestamp information will be extracted from the original CSVs using `source_file` and `source_row` mappings.
4. **Correlation**: For each flow in the filtered Friday test subset, label it as "detected by Snort" if a matching alert is found within an acceptable time window.
5. **Evaluation**: Compute the same metrics (Accuracy, Precision, Recall, F1, FPR, Confusion Matrix, per-attack recall) for Snort on this subset and compare against all three tuned ML models evaluated on the identical subset.

---

## 4. RESULTS AND DISCUSSION (ML MODELS)

### 4.1 ML vs ML Comparison (Full Test Set) - Threshold Tuned

*(See Table below - values from `reports/tuning_test/`)*

| Metric | Random Forest (t=0.6) | XGBoost (t=0.7) | Neural Network (MLP) (t=0.7) | Winner |
|---|---|---|---|---|
| Accuracy | 0.998879 | 0.999385 | 0.979771 | **XGBoost** |
| Precision | 0.997033 | 0.996926 | 0.927159 | **Random Forest** |
| Recall | 0.996318 | 0.999435 | 0.955134 | **XGBoost** |
| F1-Score | 0.996676 | 0.998179 | 0.940939 | **XGBoost** |
| Average Precision (PR-AUC) | 0.999881 | 0.999938 | 0.991683 | **XGBoost** |
| False Positive Rate | 0.000602 | 0.000625 | 0.015229 | **Random Forest** |
| Missed Attacks (FN) | 313 | 48 | 3814 | **XGBoost** |
| False Alarms (FP) | 252 | 262 | 6379 | **Random Forest** |

*Figures: `reports/figures/01_overall_comparison.png`, `reports/figures/01b_fpr_comparison.png`, `reports/figures/02_confusion_matrices.png`, `reports/figures/04_pr_curves.png`*

XGBoost (t=0.7) achieves the highest F1-score and recall whilst maintaining a very low false positive rate. Although Random Forest has slightly fewer false alarms, XGBoost misses significantly fewer attacks overall. For an IDS, this trade-off heavily favours XGBoost.

### 4.2 Per-Attack Recall Breakdown - Threshold Tuned
*(See Table below)*

| Attack | RF Recall (t=0.6) (Detected/Total) | XGBoost Recall (t=0.7) (Detected/Total) | NN Recall (t=0.7) (Detected/Total) | Notes |
|---|---|---|---|---|
| Bot | 0.61183 (238/389) | 0.94344 (367/389) | 0.36504 (142/389) | **Huge gap.** XGBoost dominates. |
| DDoS | 0.99965 (25593/25602) | 0.99977 (25596/25602) | 0.98070 (25108/25602) | Near perfect - relevant for Friday subset. |
| DoS Hulk | 0.99713 (34445/34544) | 0.99994 (34542/34544) | 0.94714 (32718/34544) | XGBoost nearly perfect. |
| DoS GoldenEye | 0.99514 (2048/2058) | 1.00000 (2058/2058) | 0.95384 (1963/2058) | XGBoost perfect. |
| PortScan | 0.99978 (18022/18026) | 0.99983 (18023/18026) | 0.96655 (17423/18026) | All strong. |
| FTP-Patator | 0.99578 (1181/1186) | 0.99747 (1183/1186) | 0.98735 (1171/1186) | All strong. |
| SSH-Patator | 0.98447 (634/644) | 0.99689 (642/644) | 0.90528 (583/644) | XGBoost leads. |
| Web Attack � Brute Force | 0.98639 (290/294) | 1.00000 (294/294) | 0.07483 (22/294) | XGboost perfect, NN struggles. |
| Web Attack � XSS | 0.96154 (125/130) | 0.99231 (129/130) | 0.00000 (0/130) | XGBoost leads, NN fails. |
| DoS Slowhttptest | 0.99426 (1040/1046) | 0.99809 (1044/1046) | 0.97897 (1024/1046) | All solid. |
| DoS slowloris | 0.99629 (1073/1077) | 0.99814 (1075/1077) | 0.96193 (1036/1077) | All solid. |
| Infiltration | 0.57143 (4/7) | 0.57143 (4/7) | 0.28571 (2/7) | Tiny sample. |
| Web Attack � Sql Injection | 0.25000 (1/4) | 0.50000 (2/4) | 0.25000 (0/0) | Tiny sample. |
| Heartbleed | 1.00000 (2/2) | 1.00000 (2/2) | 1.00000 (2/2) | Too small. |

*Figures: `reports/figures/05_xgb_per_attack_recall.png`, `reports/figures/06_per_attack_comparison.png`*

### 4.3 Selection of Champion Model
Based on the above results, **XGBoost with threshold 0.7** was selected as the primary machine learning model for comparison with Snort. This decision was based on its highest F1-score (0.998179), highest recall (0.999435), very low FPR (0.000625) and consistently strong performance across most attack classes, especially DDoS which is present in the Friday subset.

---

## 5. SNORT VS ML COMPARISON
*(To be completed once Snort evaluation is done)*

### 5.1 Snort Setup
*(Details of Snort version, ruleset, configuration and PCAP used to be added)*

### 5.2 Friday Subset Extraction
From the held-out test set, **44,669 flows** were identified from `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv`. All three tuned ML models will be re-evaluated on this identical subset for a fair comparison.

### 5.2 Correlation of Snort Alerts to Flows
*(Methodology and results of correlation to be added - include correlation success rate)*

### 5.3 Results on Friday Subset
*(Comparison table and per-attack breakdown to be added)*

### 5.4 Discussion
*(Analysis of trade-offs between Snort and ML models to be added)*

---

## 6. CONCLUSION AND FUTURE WORK
*(To be written at the end)*

### 6.1 Conclusion
*(Summary of key findings once Snort comparison is complete)*

### 6.2 Limitations
- Only a single day's PCAP (Friday) was used for the Snort comparison due to storage and download constraints.
- Correlation between Snort alerts and CICFlowMeter flows may not be perfect due to timestamp precision and flow aggregation.
- Snort performance is highly dependent on the ruleset used (Community Ruleset).
- Some attack classes have very small sample sizes (e.g. Heartbleed, Infiltration).

### 6.3 Future Work
- Evaluate Snort on multiple days of traffic for a more comprehensive comparison.
- Experiment with different Snort rulesets (e.g. Emerging Threats) to see impact on detection.
- Try threshold tuning specifically on the Friday subset.
- Explore feature importance (SHAP) for better explainability.
- Investigate deep learning approaches for improved detection of harder classes like Bot.

---

## REFERENCES
1. Sharafaldin, I., Lashkari, A. H., & Ghorbani, A. A. (2018). Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization. ICISSP 2018.
2. Canadian Institute for Cybersecurity (CIC) - CICIDS2017 Dataset. https://www.unb.ca/cic/datasets/ids-2017.html
3. Snort - Network Intrusion Detection & Prevention System. https://www.snort.org/
4. Pedregosa et al., Scikit-learn: Machine Learning in Python, JMLR 12, pp. 2825-2830, 2011.
5. Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. KDD '16.

---

## APPENDIX

### Appendix A: Project Structure
See workspace structure in project root.

### Appendix B: Key Reports
- `reports/cleaning_report.json` - Data cleaning statistics
- `reports/split_report.json` - Train/val/test split information
- `reports/tuning_test/` - Threshold-tuned model evaluation results
- `reports/initial_test/` - Baseline model evaluation results

### Appendix C: Figures
- Tuned Models: `reports/figures/`
- Baseline Models: `reports/figures_baseline/`

### Appendix D: Scripts
- `src/clean_dataset.py` - Data cleaning
- `src/create_splits.py` - Dataset splitting
- `src/train_random_forest.py`, `src/train_xgboost.py`, `src/train_neural_network.py` - Model training
- `src/evaluate_test.py` - Baseline evaluation
- `src/tune_thresholds.py` - Threshold tuning
- `src/plot_results.py`, `src/plot_results_baseline.py` - Figure generation
