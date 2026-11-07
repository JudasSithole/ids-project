# A Comparative Study of Snort Signature-Based IDS and Machine Learning-Based Intrusion Detection Systems Using the CICIDS2017 Dataset

*Student Name: [Your Full Name]*  
*Supervisor: [Supervisor Name]*  
*Department: [Department Name]*  
*Institution: [University Name]*  
*Date: [Submission Date]*

---

## Abstract

Intrusion Detection Systems (IDS) play a vital role in safeguarding network infrastructures against cyber threats. Signature-based IDS such as Snort rely on predefined attack patterns, whereas Machine Learning (ML)-based IDS can learn traffic behaviour to detect malicious flows. This paper presents a comparative evaluation of Snort and three supervised ML classifiers—Random Forest (RF), Extreme Gradient Boosting (XGBoost), and Multilayer Perceptron (MLP)—using the CICIDS2017 dataset. Flow-based features extracted by CICFlowMeter were preprocessed, cleaned, and split using stratified sampling into training (60%), validation (20%), and test (20%) sets with a fixed random seed for reproducibility. Threshold tuning was performed on the validation set to optimise the decision threshold for each model. On the full held-out test set, XGBoost with a threshold of 0.7 achieved the best overall performance with an F1-score of 0.99818, recall of 0.99944, precision of 0.99693 and a false positive rate (FPR) of 0.000625. Owing to its superior balance between detection rate and false alarms, XGBoost was selected as the primary ML model for comparison with Snort. Due to computational and storage constraints, Snort was evaluated on Friday-WorkingHours traffic from CICIDS2017. Only flows originating from the Friday PCAP that were present in the held-out test set (44,669 flows) were considered to ensure an apples-to-apples comparison. This study analyses the trade-offs between signature-based and ML-based approaches, highlighting scenarios in which each technique is more effective.

**Keywords**— Intrusion Detection System (IDS), Snort, Machine Learning, Random Forest, XGBoost, MLP, CICIDS2017, Threshold Tuning, Network Security

---

## I. Introduction

### A. Background
The rapid growth of network traffic and the increasing sophistication of cyberattacks have made network security a critical concern. Intrusion Detection Systems (IDS) are designed to monitor network traffic and identify malicious activities. Broadly, IDS are categorised as signature-based or anomaly-based. Signature-based IDS, such as Snort [3], detect attacks by matching traffic against a database of known attack signatures. While effective against known threats with low false positives, they fail to detect novel, zero-day, or obfuscated attacks. Anomaly-based approaches, particularly those utilising Machine Learning (ML), aim to learn the characteristics of benign traffic and flag deviations as potential intrusions, offering the potential to generalise to unseen attacks [1], [4].

### B. Motivation
Most prior works evaluate Snort and ML-based IDS independently. A fair comparative study requires evaluating both on identical traffic with identical ground truth. The CICIDS2017 dataset [1], [2] provides realistic benign and attack traffic, making it suitable for such an evaluation. However, comparing the two approaches is non-trivial, as Snort inspects raw packets whilst ML models trained on CICFlowMeter-extracted flow features operate at the flow level.

### C. Problem Statement
This work addresses the question: *"For the same network traffic, how do the detection capabilities of Snort (signature-based) compare with flow-based Machine Learning models (Random Forest, XGBoost, MLP)?"*

### D. Aim and Objectives
**Aim:** To conduct a fair comparison between Snort and selected ML-based classifiers for network intrusion detection using the CICIDS2017 dataset.

**Objectives:**
1) Preprocess and clean the CICIDS2017 flow dataset.
2) Perform stratified train/validation/test splitting with reproducibility controls.
3) Train and evaluate RF, XGBoost, and MLP classifiers using standard IDS metrics.
4) Apply threshold tuning on the validation set to optimise decision boundaries.
5) Evaluate Snort on a representative CICIDS2017 PCAP and correlate alerts to flows.
6) Compare Snort against the tuned ML models on the identical traffic subset.

### E. Scope and Limitations
Due to time and storage constraints, only Friday-WorkingHours traffic is used for the Snort–ML comparison. Only flows from this capture that exist within the held-out test set are considered. Snort evaluation is conducted using the Snort Community Ruleset. Correlation between Snort alerts and CICFlowMeter flows is subject to limitations imposed by packet aggregation and timestamp precision.

---

## II. Related Work
*(To be expanded. Cite relevant studies comparing Snort/Suricata with ML on CICIDS2017 and discuss the observed trade-offs between signature-based and learning-based approaches.)*

---

## III. Methodology

### A. Dataset
CICIDS2017 [1], [2] was utilised. It comprises benign traffic and various attack types captured over five days. Eight CSV files generated by CICFlowMeter were used for ML training and evaluation. 

### B. Data Preprocessing
Preprocessing was implemented in `src/clean_dataset.py`:
1) Stripping of whitespace from column names
2) Validation and removal of records with missing/empty labels
3) Verification and removal of the duplicate header column `Fwd Header Length.1`
4) Coercion to numeric values with non-finite row removal
5) Quarantine of identical feature vectors with conflicting binary labels
6) Removal of exact duplicate flows with identical binary targets (first occurrence retained)
7) Assignment of binary label `target` (1 = Attack, 0 = Benign) with preservation of `original_label` for per-attack analysis

After preprocessing, 2,519,404 flows remained with 77 features. Full cleaning statistics are recorded in `reports/cleaning_report.json`.

### C. Dataset Splitting
Stratified splitting by `original_label` was performed using `scikit-learn` [4] with `random_state=42` to obtain Train (60%), Validation (20%), and Test (20%) sets. Split indices were stored in `data/processed/split_indices.npz`. The SHA-256 hash of `cleaned_flows.pkl` was logged in `reports/split_report.json` and verified by all downstream scripts to enforce reproducibility.

### D. Machine Learning Models

#### 1) Random Forest
`RandomForestClassifier` [4] with `n_estimators=100`, `max_depth=16`, `min_samples_leaf=5`, `max_features='sqrt'`, `class_weight='balanced'`, `n_jobs=2`, `random_state=42`.

#### 2) XGBoost
`XGBClassifier` [5] with `tree_method='hist'`, `device='cpu'`, `n_estimators=200`, `max_depth=6`, `learning_rate=0.1`, `min_child_weight=5`, `subsample=0.8`, `colsample_bytree=0.8`, `reg_lambda=1.0`, `scale_pos_weight = N_benign/N_attack`, `n_jobs=2`, `random_state=42`. Saved as `models/xgboost_baseline.ubj`.

#### 3) Multilayer Perceptron
`MLPClassifier` [4] with hidden layers (32,16), ReLU, Adam. `StandardScaler` was fit on the training set only. A custom mini-batch training loop with `partial_fit`, manual shuffling per epoch, class-weighted sample weights, validation-based early stopping (patience=4), and best-model checkpointing was implemented (batch size=4096, max epochs=20).

### E. Threshold Tuning
To better reflect operational IDS trade-offs, decision thresholds were tuned on the validation set by sweeping τ ∈ {0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7}. For each model, the threshold maximising validation F1-score was selected and evaluated on the locked test set. Selected thresholds: RF = 0.6, XGBoost = 0.7, MLP = 0.7.

### F. Evaluation Metrics
The following metrics were used: Accuracy, Precision, Recall (TPR), F1-score, False Positive Rate (FPR = FP/(FP+TN)), Average Precision (AP/PR-AUC), Confusion Matrix, and per-attack recall. Due to class imbalance, Precision-Recall analysis was emphasised alongside ROC analysis.

### G. Snort Evaluation Approach
1) **PCAP Selection:** Friday-WorkingHours PCAP containing DDoS traffic. 
2) **Ruleset:** Snort Community Ruleset (version to be recorded).
3) **Execution:** Snort run in IDS mode against the selected PCAP with alerts logged to fast.log/alert output.
4) **Subset Alignment:** Filter test set to flows with `source_file = Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv` present in the held-out test split (n = 44,669).
5) **Correlation:** Map Snort alerts to flows using 5-tuple (Src IP, Dst IP, Src Port, Dst Port, Protocol) and timestamp proximity, leveraging IP/Port/Timestamp from original CSVs where available.
6) **Comparative Evaluation:** Compute identical metrics for Snort on the correlated subset and compare against all tuned ML models evaluated on the same subset.

---

## IV. Results and Discussion

### A. ML vs ML Comparison on Full Test Set (Threshold Tuned)

Table I presents the performance of all three tuned models on the full held-out test set.

**TABLE I**  
**PERFORMANCE COMPARISON OF TUNED ML MODELS ON FULL TEST SET**

| Metric | Random Forest (τ=0.6) | XGBoost (τ=0.7) | MLP (τ=0.7) | Best |
|---|---|---|---|---|
| Accuracy | 0.998879 | 0.999385 | 0.979771 | XGBoost |
| Precision | 0.997033 | 0.996926 | 0.927159 | Random Forest |
| Recall | 0.996318 | 0.999435 | 0.955134 | XGBoost |
| F1-Score | 0.996676 | 0.998179 | 0.940939 | XGBoost |
| Average Precision (PR-AUC) | 0.999881 | 0.999938 | 0.991683 | XGBoost |
| False Positive Rate (FPR) | 0.000602 | 0.000625 | 0.015229 | Random Forest |
| Missed Attacks (FN) | 313 | 48 | 3814 | XGBoost |
| False Alarms (FP) | 252 | 262 | 6379 | Random Forest |

Fig. 1 shows the overall performance comparison. Fig. 2 depicts the confusion matrices. Fig. 3 and Fig. 4 show the ROC and Precision-Recall curves respectively. XGBoost (τ=0.7) offers the best balance with the highest F1 and recall whilst maintaining a very low FPR.

### B. Per-Attack Detection Performance

Table II shows per-attack recall for all three tuned models.

**TABLE II**  
**PER-ATTACK RECALL COMPARISON (TUNED MODELS)**

| Attack | RF (τ=0.6) | XGBoost (τ=0.7) | MLP (τ=0.7) |
|---|---|---|---|
| Bot | 0.61183 (238/389) | 0.94344 (367/389) | 0.36504 (142/389) |
| DDoS | 0.99965 (25593/25602) | 0.99977 (25596/25602) | 0.98070 (25108/25602) |
| DoS Hulk | 0.99713 (34445/34544) | 0.99994 (34542/34544) | 0.94714 (32718/34544) |
| DoS GoldenEye | 0.99514 (2048/2058) | 1.00000 (2058/2058) | 0.95384 (1963/2058) |
| DoS Slowhttptest | 0.99426 (1040/1046) | 0.99809 (1044/1046) | 0.97897 (1024/1046) |
| DoS slowloris | 0.99629 (1073/1077) | 0.99814 (1075/1077) | 0.96193 (1036/1077) |
| FTP-Patator | 0.99578 (1181/1186) | 0.99747 (1183/1186) | 0.98735 (1171/1186) |
| SSH-Patator | 0.98447 (634/644) | 0.99689 (642/644) | 0.90528 (583/644) |
| PortScan | 0.99978 (18022/18026) | 0.99983 (18023/18026) | 0.96655 (17423/18026) |
| Web Attack – Brute Force | 0.98639 (290/294) | 1.00000 (294/294) | 0.07483 (22/294) |
| Web Attack – XSS | 0.96154 (125/130) | 0.99231 (129/130) | 0.00000 (0/130) |
| Infiltration | 0.57143 (4/7) | 0.57143 (4/7) | 0.28571 (2/7) |
| Web Attack – Sql Injection | 0.25000 (1/4) | 0.50000 (2/4) | 0.25000 (1/4) |
| Heartbleed | 1.00000 (2/2) | 1.00000 (2/2) | 1.00000 (2/2) |

Fig. 5 shows XGBoost per-attack recall, whilst Fig. 6 compares all three models per-attack. XGBoost significantly outperforms RF and MLP on Bot, SSH-Patator, Web Attacks and shows near-perfect detection on DDoS and DoS variants.

### C. Selection of Champion Model
Based on overall F1-score, recall, and per-attack performance, **XGBoost with threshold 0.7** was selected as the primary ML model for the Snort comparison.

### D. Snort vs ML Comparison (Friday Subset)
*(To be completed)*

**TABLE III**  
**COMPARISON ON FRIDAY SUBSET (SNORT VS TUNED ML MODELS)**

*(Results to be inserted once correlation is completed)*

**TABLE IV**  
**PER-ATTACK COMPARPARISON ON FRIDAY SUBSET**

*(Results to be inserted)*

Fig. 7: Snort vs ML comparison on Friday subset *(to be added)*

---

## V. Conclusion and Future Work

### A. Conclusion
*(To be written after completing Section IV.D)*

### B. Limitations
1. Only Friday-WorkingHours traffic was used for the Snort comparison due to PCAP availability/size constraints.
2. Correlation between Snort alerts and CICFlowMeter flows may not achieve 100% mapping due to aggregation and timing.
3. Snort detection is limited to attacks covered by the Community Ruleset.
4. Flow-based features do not include raw payload inspection in the same form as Snort.
5. Some attack classes have minimal support in the subset, limiting statistical significance.

### C. Future Work
1. Extend the comparison to multiple CICIDS2017 days/PCAPs.
2. Evaluate with alternative rulesets (e.g., Emerging Threats).
3. Investigate improved alert-flow correlation techniques.
4. Apply explainability methods (e.g., SHAP) to analyse ML decisions.
5. Explore hybrid approaches combining signature-based and ML-based detection.

---

## References

[1] I. Sharafaldin, A. H. Lashkari, and A. A. Ghorbani, "Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization," in *Proc. 4th International Conference on Information Systems Security and Privacy (ICISSP)*, 2018, pp. 108–116.

[2] Canadian Institute for Cybersecurity, "CICIDS2017 Dataset," University of New Brunswick. [Online]. Available: https://www.unb.ca/cic/datasets/ids-2017.html

[3] M. Roesch, "Snort - Lightweight Intrusion Detection for Networks," in *Proc. 13th USENIX LISA Conference*, 1999, pp. 229–238.

[4] F. Pedregosa et al., "Scikit-learn: Machine Learning in Python," *J. Mach. Learn. Res.*, vol. 12, pp. 2825–2830, 2011.

[5] T. Chen and C. Guestrin, "XGBoost: A Scalable Tree Boosting System," in *Proc. 22nd ACM SIGKDD*, 2016, pp. 785–794.

---

## Appendix

### A. Figures
- Fig. 1: `reports/figures/01_overall_comparison.png`
- Fig. 2: `reports/figures/02_confusion_matrices.png`
- Fig. 3: `reports/figures/03_roc_curves.png`
- Fig. 4: `reports/figures/04_pr_curves.png`
- Fig. 5: `reports/figures/05_xgb_per_attack_recall.png`
- Fig. 6: `reports/figures/06_per_attack_comparison.png`
- Fig. 7: *(To be added after Snort comparison)*

### B. Reproducibility Artifacts
- Data cleaning: `reports/cleaning_report.json`
- Splits: `reports/split_report.json`, `data/processed/split_indices.npz`
- Baseline results: `reports/initial_test/`
- Tuned results: `reports/tuning_test/`
- Models: `models/` (RF joblib, XGBoost ubj, NN joblib with scaler)
- Scripts: `src/` (all preprocessing, training, tuning, evaluation, plotting)
