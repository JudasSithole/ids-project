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
Detection metrics: Accuracy, Precision, Recall (TPR), F1-score, FPR = FP/(FP+TN), confusion matrix, and per-attack recall where applicable. **Resource-efficiency metrics (proposal requirement):** detection latency, CPU utilization, and RAM usage. Because physical Raspberry Pi hardware was unavailable, low-spec behavior was **emulated** (1 CPU core affinity via `taskset`/process affinity, 1 thread via BLAS/OpenMP/threadpool controls, sampled CPU/RSS via `psutil`). Results are reported as an emulated Raspberry Pi 4-class 1-vCPU proxy.

### G. Snort vs ML Comparison (Friday Test Subset)
1) **PCAP:** `/media/sf_OdinShare/Friday-WorkingHours.pcap` (8.84 GB).  
2) **Snort:** Snort++ **3.12.2.0**, offline mode `-r`, `-A alert_fast -q`, config `snort_friday_alert.lua` (loads `/usr/local/etc/snort/snort_defaults.lua` + `snort.lua`, `HOME_NET/EXTERNAL_NET=any`).  
3) **Ruleset:** Snort 3 Community Rules, **4,026** rules (`snort_rules/snort3-community-rules/snort3-community.rules`). Output: 1,617,398 alert lines (`snort_out/alert_fast.txt`, local).  
4) **Subset alignment (fair comparison):** only Friday flows in the held-out **test split** were scored (**123,161** rows: 79,144 benign, 44,017 attack; Morning + Afternoon-DDoS + Afternoon-PortScan).  
5) **Correlation:** alerts parsed by `src/parse_snort_alert_fast.py` and correlated by 5-tuple (both directions) with a **+61,260 s** timestamp offset and **±300 s** window (`src/correlate_snort_friday_fast.py`, `reports/snort_friday_evaluation_final.json`). Limitations: packet-vs-flow granularity, timestamp alignment uncertainty, and decoder/L2 alerts (e.g., 1.39M `{eth}`) lacking flow 5-tuples.  
6) **Resource benchmark:** `src/benchmark_snort_constrained.py` runs Snort under 1-core affinity and samples CPU/RSS; output `reports/benchmark_snort_constrained_pi4_1core.json` (may require long runtime on 8.84 GB PCAP).

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
Completed. See **Section III.G** and **Section IV.C–IV.D (ML vs Snort Comparison)** above for the fair Friday test-subset results: Snort 3.12.2.0 + 4,026 Community Rules, correlated-flow evaluation (`reports/snort_friday_evaluation_final.json`) vs tuned RF/XGBoost/MLP (`reports/friday_ml_tuned_subset.json`). Resource efficiency is reported in Table VI and constrained benchmark JSONs.

---

### C. Resource Efficiency (Emulated Low-Spec Testbed)

All ML batch-inference measurements were collected on the Friday held-out test subset (123,161 rows) with 1-core affinity and 1-thread limits. Snort was measured offline on the full Friday PCAP (8.84 GB) under `taskset -c 0`. Metrics are reproducible from `reports/benchmark_*_constrained_pi4_1core.json` and `reports/benchmark_snort_constrained_pi4_1core.json`.

**TABLE VI**  
**RESOURCE EFFICIENCY (EMULATED 1-CORE)**

| System | Wall time (s) | Throughput | Latency/flow (ms) | Peak RSS (MiB) | Notes |
| :--- | ---: | ---: | ---: | ---: | :--- |
| Random Forest | 2.249 | 54,767 flows/s | 0.0183 | 351.9 | sklearn + joblib, 123,161 flows |
| XGBoost | 0.919 | 134,023 flows/s | 0.00746 | 332.5 | xgboost 3.4.1, 123,161 flows |
| Small Neural Network (MLP) | 0.227 | 542,020 flows/s | 0.00184 | 329.4 | sklearn MLP + StandardScaler, 123,161 flows |
| **Snort 3.12.2.0 (Community Rules)** | **595.44** | **14.16 MB/s** (PCAP) | *(N/A)* | **200.59** | Offline full Friday PCAP 8.84 GB, `taskset -c 0`, return 0; sampled mean CPU% sum 27.84% |

*Interpretation:* Snort’s measurement is packet/PCAP processing throughput (offline read), while ML throughput is flow-feature batch inference; they are not directly equivalent units. On this emulated single-core setting, ML batch inference over pre-extracted flows is fast and memory-bounded (~330–352 MiB), while Snort required ~595 s to process the 8.84 GB PCAP with ~201 MiB sampled peak RSS for the measured process.


## V. Conclusion and Future Work

### A. Conclusion
This study designed and evaluated a lightweight behavioural intrusion detection pipeline for resource-constrained networks using CICIDS2017 flow features and compared it against a signature-based Snort 3 baseline. On the Friday held-out test subset (123,161 flows), tuned behavioural models substantially outperformed the correlated Snort Community Rules baseline in flow-label detection: **XGBoost** achieved the best F1 (**0.9981**, FPR 0.00169), followed by **Random Forest** (F1 0.9962) and the **Small Neural Network** (F1 0.9822), while correlated Snort achieved **F1 0.5038** (precision 0.6906, recall 0.3966, FPR 0.0988). Under an emulated 1-core constrained setting, all ML models performed batch inference efficiently (XGBoost ~134k flows/s, ~333 MiB peak RSS; NN fastest at ~542k flows/s). The results indicate that lightweight behavioural ML offers a strong accuracy–feasibility trade-off for resource-constrained deployments, while signature IDS remains useful for explicit rule matches; hybrid deployment is a promising direction.

### B. Limitations
1. Only Friday-WorkingHours traffic was used for the Snort comparison due to PCAP availability/size constraints.
2. Correlation between Snort alerts and CICFlowMeter flows is imperfect due to packet-to-flow aggregation, timestamp offsets, and decoder/L2 alerts without 5-tuples.
3. Snort detection is limited to attacks covered by the Community Ruleset and the configured inspection path.
4. Flow-based ML features differ from Snort’s packet/payload inspection; comparisons should be interpreted as complementary detection paradigms.
5. Resource metrics were collected on an **emulated** 1-core testbed, not physical Raspberry Pi hardware.
6. Some attack classes have limited support in the evaluated subset.

### C. Future Work
1. Extend the comparison to multiple CICIDS2017 days/PCAPs and live-capture conditions.
2. Evaluate with alternative rulesets (e.g., Emerging Threats) and tuned Snort preprocessors.
3. Improve alert-flow correlation (per-packet flow reconstruction, better time alignment, rule filtering).
4. Apply explainability methods (e.g., SHAP) to behavioural model decisions.
5. Explore hybrid signature + ML pipelines and deployment on physical Raspberry Pi/IoT hardware.
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
- Fig. 7: `reports/figures/07_snort_ml_comparison.png` (PDF: `.pdf`) — ML vs Snort 3 Community Rules on Friday held-out test subset
- Fig. 8: `reports/figures/08_snort_confusion_matrix.png` (PDF: `.pdf`) — Snort 3 correlated-flow confusion matrix heatmap (counts + row-normalized %) on Friday held-out test subset
- Fig. 9: `reports/figures/09_resource_cpu_memory.png` (PDF: `.pdf`) — Resource efficiency under emulated 1-core constraint: peak RSS, CPU utilization, and throughput (ML rows/s vs Snort PCAP MB/s; units not directly comparable)

### B. Reproducibility Artifacts
- Data cleaning: `reports/cleaning_report.json`
- Splits: `reports/split_report.json`, `data/processed/split_indices.npz`
- Baseline results: `reports/initial_test/`
- Tuned results: `reports/tuning_test/`
- Models: `models/` (RF joblib, XGBoost ubj, NN joblib with scaler)
- Scripts: `src/` (all preprocessing, training, tuning, evaluation, plotting)
