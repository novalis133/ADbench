// Full data from both `algorithms.md` and `datasets.md`
// Data for algorithms and datasets is structured below
const data = [
    // Algorithms from algorithms.md
    // 1. Unsupervised Anomaly Detection Algorithms
    
    {
        category: "Unsupervised",
        name: "Isolation Forest (IForest)",
        description: "Isolates anomalies by recursively partitioning the data using randomly selected features and thresholds.",
        useCases: ["Fraud detection", "Network intrusion detection", "Cybersecurity"],
        link: "https://ieeexplore.ieee.org/document/4781136",
        citation: "Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008). Isolation forest. Data Mining and Knowledge Discovery, 20(2), 272-292."
    },
    {
        category: "Unsupervised",
        name: "LOF (Local Outlier Factor)",
        description: "Detects outliers by comparing the local density of a point to that of its neighbors.",
        useCases: ["Intrusion detection", "Healthcare anomaly detection", "Fraud detection"],
        link: "https://dl.acm.org/doi/10.1145/342009.335388",
        citation: "Breunig, M. M., Kriegel, H.-P., Ng, R. T., & Sander, J. (2000). LOF: Identifying density-based local outliers."
    },
    {
        category: "Unsupervised",
        name: "KNN (K-Nearest Neighbors)",
        description: "Identifies outliers by measuring the distance between a point and its k-nearest neighbors.",
        useCases: ["Network intrusion detection", "Fraud detection"],
        link: "https://www.researchgate.net/profile/Clara-Pizzuti/publication/220699183_Fast_Outlier_Detection_in_High_Dimensional_Spaces/links/542ea6a60cf27e39fa9635c6/Fast-Outlier-Detection-in-High-Dimensional-Spaces.pdf",
        citation: "Angiulli, F., & Pizzuti, C. (2006). Fast outlier detection in high dimensional spaces."
    },
    {
        category: "Unsupervised",
        name: "COPOD (Copula-Based Outlier Detection)",
        description: "Uses copula theory to model the joint distribution of data and detect anomalies based on the joint cumulative distribution function.",
        useCases: ["High-dimensional anomaly detection", "Financial anomaly detection"],
        link: "https://ieeexplore.ieee.org/document/9338429",
        citation: "Li, Y., Zhao, Z., & Botta, N. (2020). COPOD: Copula-based outlier detection."
    },
    {
        category: "Unsupervised",
        name: "ECOD (Empirical CDF-Based Outlier Detection)",
        description: "Non-parametric outlier detection method that uses empirical cumulative distribution functions.",
        useCases: ["Large-scale anomaly detection", "Real-time sensor monitoring"],
        link: "https://arxiv.org/abs/2201.00382",
        citation: "Li, Z., Zhao, Z., Botta, N., Ifrim, G., & Stolfo, S. (2020). ECOD: Unsupervised outlier detection using empirical cumulative distribution functions."
    },
    {
        category: "Unsupervised",
        name: "DeepSVDD (Deep Support Vector Data Description)",
        description: "Maps data into a latent space where normal data points are compact, and anomalies lie outside the compact region.",
        useCases: ["Network intrusion detection", "Image anomaly detection"],
        link: "https://ml.cs.uni-kl.de/publications/2018/deep-svdd.pdf",
        citation: "Ruff, L., Vandermeulen, R. A., Görnitz, N., Deecke, L., Siddiqui, S. A., Binder, A., Müller, K. R., & Kloft, M. (2018). Deep one-class classification."
    },
    {
        category: "Unsupervised",
        name: "DAGMM (Deep Autoencoding Gaussian Mixture Model)",
        description: "Combines an autoencoder for dimensionality reduction with a Gaussian Mixture Model (GMM) for probabilistic modeling.",
        useCases: ["Industrial monitoring", "Fraud detection", "Financial data anomaly detection"],
        link: "https://openreview.net/forum?id=BJJLHbb0-",
        citation: "Zong, B., Song, Q., Qi, Y., Huang, X., & Dhillon, I. (2018). Deep autoencoding Gaussian mixture model for unsupervised anomaly detection."
    },
    {
        category: "Unsupervised",
        name: "Histogram-Based Outlier Score (HBOS)",
        description: "Uses histograms to model feature distributions and detect outliers.",
        useCases: ["Anomaly detection in high-dimensional data"],
        link: "https://www.goldiges.de/publications/HBOS-KI-2012.pdf",
        citation: "Goldstein, M., & Dengel, A. (2012). Histogram-based outlier score (HBOS)."
    },
    {
        category: "Unsupervised",
        name: "Principal Component Analysis (PCA)",
        description: "Detects anomalies by finding points that cannot be well represented by the principal components.",
        useCases: ["Dimensionality reduction for anomaly detection"],
        link: "https://www.tandfonline.com/doi/abs/10.1080/14786440109462720",
        citation: "Pearson, K. (1901). On lines and planes of closest fit to systems of points in space."
    },
    {
        category: "Unsupervised",
        name: "Minimum Covariance Determinant (MCD)",
        description: "Fits an ellipse to the data distribution and detects outliers.",
        useCases: ["High-dimensional anomaly detection"],
        link: "https://www.jstor.org/stable/2288718",
        citation: "Rousseeuw, P. J. (1984). Least median of squares regression."
    },
    {
        category: "Unsupervised",
        name: "Connectivity-Based Outlier Factor (COF)",
        description: "Measures isolation by calculating how well-connected a point is to its neighbors.",
        useCases: ["Low-density outlier detection"],
        link: "http://www.cse.cuhk.edu.hk/~adafu/Pub/pakdd02.pdf",
        citation: "Tang, J., Chen, Z., Fu, A. W.-C., & Cheung, D. W. (2002). Enhancing effectiveness of outlier detections for low-density patterns."
    },
    {
        category: "Unsupervised",
        name: "Subspace Outlier Detection (SOD)",
        description: "Identifies points that are isolated in some subspace of the data.",
        useCases: ["High-dimensional data anomaly detection"],
        link: "https://www.dbs.ifi.lmu.de/Publikationen/Papers/pakdd09_SOD.pdf",
        citation: "Kriegel, H.-P., Kröger, P., Schubert, E., & Zimek, A. (2009). Outlier detection in axis-parallel subspaces of high-dimensional data."
    },
    {
        category: "Unsupervised",
        name: "Feature Bagging (FB)",
        description: "Uses bagging techniques with different feature subsets to detect anomalies.",
        useCases: ["Ensemble-based anomaly detection"],
        link: "https://dl.acm.org/doi/10.1145/1081870.1081891",
        citation: "Lazarevic, A., & Kumar, V. (2005). Feature bagging for outlier detection."
    },
    {
        category: "Unsupervised",
        name: "Robust Principal Component Analysis (RPCA)",
        description: "Decomposes data into a low-rank component and a sparse component, with anomalies in the sparse component.",
        useCases: ["Anomaly detection in noisy datasets"],
        link: "https://dl.acm.org/doi/10.1145/1970392.1970395",
        citation: "Candès, E. J., Li, X., Ma, Y., & Wright, J. (2010). Robust principal component analysis."
    },

    // Semi-supervised Anomaly Detection Algorithms
    {
        category: "Semi-supervised",
        name: "DeepSAD (Deep Semi-supervised Anomaly Detection)",
        description: "Combines labeled and unlabeled data for semi-supervised anomaly detection by learning compact representations of normal data.",
        useCases: ["Fraud detection", "Medical anomaly detection", "Network security"],
        link: "https://ml.cs.uni-kl.de/publications/2020/deep_semi_supervised_anomaly_detection.pdf",
        citation: "Ruff, L., Kauffmann, J. R., Vandermeulen, R., Görnitz, N., Deecke, L., & Kloft, M. (2020). Deep semi-supervised anomaly detection."
    },
    {
        category: "Semi-supervised",
        name: "REPEN (Representation Learning-based Anomaly Detection)",
        description: "Learns a low-dimensional representation of the data and detects anomalies by measuring distances in this learned space.",
        useCases: ["Financial fraud detection", "Sensor data anomaly detection"],
        link: "https://arxiv.org/pdf/1806.04808",
        citation: "Pang, G., Cao, L., Chen, L., & Liu, H. (2017). Learning representations of ultrahigh-dimensional data for random distance-based outlier detection."
    },
    {
        category: "Semi-supervised",
        name: "GANomaly (Deep Generative Adversarial Networks for Anomaly Detection)",
        description: "Uses GANs to generate synthetic normal data, with anomalies detected as poorly reconstructed data points.",
        useCases: ["Industrial anomaly detection", "Fraud detection"],
        link: "https://arxiv.org/abs/1805.06725",
        citation: "Akcay, S., Atapour-Abarghouei, A., & Breckon, T. P. (2018). GANomaly: Semi-supervised anomaly detection via adversarial training."
    },
    {
        category: "Semi-supervised",
        name: "DevNet (Deep Anomaly Detection with Deviations)",
        description: "Learns deviations from normal data using limited labeled anomalies and a large number of unlabeled data points.",
        useCases: ["Fraud detection", "Network security anomaly detection"],
        link: "https://arxiv.org/abs/1911.08623",
        citation: "Pang, G., Shen, C., Cao, L., van den Hengel, A., & Liu, W. (2020). Deep anomaly detection with deviation networks."
    },
    {
        category: "Semi-supervised",
        name: "RDP (Robust Deep PCA)",
        description: "Combines deep learning with PCA to create robust representations of data for detecting anomalies in noisy or high-dimensional datasets.",
        useCases: ["Anomaly detection in noisy datasets", "Financial fraud detection"],
        link: "https://arxiv.org/abs/2208.01998",
        citation: "Hong-Lan Botterman, Julien Roussel, Thomas Morzadec, Ali Jabbari, Nicolas Brunel. Robust PCA for Anomaly Detection and Data Imputation in Seasonal Time Series."
    },
    {
        category: "Semi-supervised",
        name: "SO-GAAL (Single Objective Generative Adversarial Active Learning)",
        description: "Generates anomalies using GANs and uses active learning to detect real anomalies.",
        useCases: ["Image anomaly detection", "Fraud detection"],
        link: "https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8668550",
        citation: "Liu, F. T., Ting, K. M., & Zhou, Z.-H. (2019). Generative adversarial active learning for anomaly detection."
    },
    {
        category: "Semi-supervised",
        name: "FEAWAD (Feature Engineering and Weak Anomaly Detection)",
        description: "Uses feature engineering combined with weak supervision to detect anomalies, typically in noisy or complex datasets.",
        useCases: ["Fraud detection", "Sensor data anomaly detection"],
        link: "https://arxiv.org/abs/2105.10500",
        citation: "Yingjie Zhou, Xucheng Song, Yanru Zhang, Fanxing Liu, Ce Zhu, Lingqiao Liu. Feature encoding with autoencoders for weakly supervised anomaly detection."
    },

    // Supervised Anomaly Detection Algorithms
    {
        category: "Supervised",
        name: "XGBoost",
        description: "A scalable, efficient gradient boosting method used for classification and regression, applied to anomaly detection as a binary classification problem.",
        useCases: ["Credit card fraud detection", "Financial anomaly detection"],
        link: "https://dl.acm.org/doi/10.1145/2939672.2939785",
        citation: "Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system."
    },
    {
        category: "Supervised",
        name: "LightGBM",
        description: "A highly efficient gradient boosting framework optimized for large datasets and faster training.",
        useCases: ["Large-scale anomaly detection", "Fraud detection"],
        link: "https://dl.acm.org/doi/10.5555/3294996.3295074",
        citation: "Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T.-Y. (2017). LightGBM: A highly efficient gradient boosting decision tree."
    },
    {
        category: "Supervised",
        name: "CatBoost",
        description: "Gradient boosting method designed to handle categorical features efficiently, often used for classification tasks in anomaly detection.",
        useCases: ["Fraud detection", "Customer churn prediction"],
        link: "https://arxiv.org/abs/1706.09516",
        citation: "Dorogush, A. V., Ershov, V., & Gulin, A. (2018). CatBoost: Gradient boosting with categorical features support."
    },
    {
        category: "Supervised",
        name: "Random Forest",
        description: "An ensemble method that combines multiple decision trees to improve classification accuracy.",
        useCases: ["Financial fraud detection", "Healthcare anomaly detection"],
        link: "https://doi.org/10.1023/A:1010933404324",
        citation: "Breiman, L. (2001). Random forests."
    },
    {
        category: "Supervised",
        name: "Multilayer Perceptron (MLP)",
        description: "A type of feedforward artificial neural network used for classification tasks, including anomaly detection.",
        useCases: ["Credit scoring", "Fraud detection"],
        link: "https://www.semanticscholar.org/paper/Learning-representations-by-back-propagating-errors-Rumelhart-Hinton/052b1d8ce63b07fec3de9dbb583772d860b7c769",
        citation: "Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning representations by back-propagating errors."
    },
    {
        category: "Supervised",
        name: "ResNet",
        description: "A deep neural network with residual connections, making it easier to train deep networks, commonly used in image-based anomaly detection.",
        useCases: ["Image anomaly detection", "Medical imaging"],
        link: "https://doi.org/10.1109/CVPR.2016.90",
        citation: "He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition."
    },
    {
        category: "Supervised",
        name: "FTTransformer",
        description: "A transformer-based model designed for tabular data, used for supervised anomaly detection in high-dimensional datasets.",
        useCases: ["Financial anomaly detection", "Sensor data analysis"],
        link: "https://arxiv.org/abs/2106.11959",
        citation: "Gorishniy, Y., Rubachev, I., Khrulkov, V., & Babenko, A. (2021). Revisiting deep learning models for tabular data."
    },
    {
        category: "Supervised",
        name: "Naive Bayes",
        description: "A probabilistic classifier that uses Bayes’ theorem with strong independence assumptions between features.",
        useCases: ["Text anomaly detection", "Fraud detection"],
        link: "https://doi.org/10.1023/A:1007413511361",
        citation: "Domingos, P., & Pazzani, M. (1997). On the optimality of the simple Bayesian classifier under zero-one loss."
    },
    {
        category: "Supervised",
        name: "Logistic Regression",
        description: "A statistical model used for binary classification problems, including anomaly detection.",
        useCases: ["Financial fraud detection", "Healthcare anomaly detection"],
        link: "https://doi.org/10.1111/j.2517-6161.1958.tb00292.x",
        citation: "Cox, D. R. (1958). The regression analysis of binary sequences."
    },
    
    // Datasets from datasets.md
        // 1. Public Benchmark Datasets
    {
        category: "Public Benchmark Datasets",
        name: "Annthyroid",
        dataType: ["tabular"],
        size: "medium",
        anomalyRatio: "high",
        description: "Medical dataset for thyroid disease detection (~21% anomalies).",
        link: "https://archive.ics.uci.edu/ml/datasets/Thyroid+Disease"
    },
    {
        category: "Public Benchmark Datasets",
        name: "Arrhythmia",
        dataType: ["tabular"],
        size: "medium",
        anomalyRatio: "medium",
        description: "Cardiac arrhythmia detection (~15% anomalies).",
        link: "https://archive.ics.uci.edu/ml/datasets/arrhythmia"
    },
    {
        category: "Public Benchmark Datasets",
        name: "Cardiotocography",
        dataType: ["tabular"],
        size: "medium",
        anomalyRatio: "medium",
        description: "Fetal health classification (~22% anomalies).",
        link: "https://archive.ics.uci.edu/ml/datasets/cardiotocography"
    },
    {
        category: "Public Benchmark Datasets",
        name: "KDDCup99",
        dataType: ["tabular"],
        size: "large",
        anomalyRatio: "low",
        description: "Network intrusion detection dataset (~3.92% anomalies).",
        link: "https://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html"
    },
    {
        category: "Public Benchmark Datasets",
        name: "Satellite",
        dataType: ["tabular"],
        size: "large",
        anomalyRatio: "medium",
        description: "Satellite image classification (~31% anomalies).",
        link: "https://archive.ics.uci.edu/ml/datasets/Statlog+(Landsat+Satellite)"
    },
    {
        category: "Public Benchmark Datasets",
        name: "Shuttle",
        dataType: ["tabular"],
        size: "large",
        anomalyRatio: "low",
        description: "NASA dataset for shuttle failure classification (~7.15% anomalies).",
        link: "https://archive.ics.uci.edu/ml/datasets/Statlog+(Shuttle)"
    },
    {
        category: "Public Benchmark Datasets",
        name: "Lymphography",
        dataType: ["tabular"],
        size: "medium",
        anomalyRatio: "medium",
        description: "Medical dataset used in anomaly detection.",
        link: "https://archive.ics.uci.edu/ml/datasets/Lymphography"
    },
    {
        category: "Public Benchmark Datasets",
        name: "Pendigits",
        dataType: ["tabular"],
        size: "large",
        anomalyRatio: "low",
        description: "Handwritten digit dataset (~2.27% anomalies).",
        link: "https://archive.ics.uci.edu/dataset/81/pen+based+recognition+of+handwritten+digits"
    },
    {
        category: "Public Benchmark Datasets",
        name: "Musk",
        dataType: ["tabular"],
        size: "medium",
        anomalyRatio: "low",
        description: "Detects if a molecule is 'musky' (~3.17% anomalies).",
        link: "https://archive.ics.uci.edu/ml/datasets/Musk+(Version+2)"
    },
    {
        category: "Public Benchmark Datasets",
        name: "Mammography",
        dataType: ["tabular"],
        size: "large",
        anomalyRatio: "low",
        description: "Breast cancer detection (~2.32% anomalies).",
        link: "https://archive.ics.uci.edu/ml/datasets/Mammographic+Mass"
    },
    {
        category: "Public Benchmark Datasets",
        name: "Breastw",
        dataType: ["tabular"],
        size: "large",
        anomalyRatio: "high",
        description: "Wisconsin breast cancer dataset (~34.47% anomalies).",
        link: "https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)"
    },
    {
        category: "Public Benchmark Datasets",
        name: "Pima",
        dataType: ["tabular"],
        size: "medium",
        anomalyRatio: "medium",
        description: "Diabetes detection (~35% anomalies).",
        link: "https://www.dbs.ifi.lmu.de/research/outlier-evaluation/DAMI/semantic/Pima/Pima_35.html"
    },
    {
        category: "Public Benchmark Datasets",
        name: "Ionosphere",
        dataType: ["tabular"],
        size: "medium",
        anomalyRatio: "medium",
        description: "Radar data for classifying ionosphere structure (~35% anomalies).",
        link: "https://archive.ics.uci.edu/ml/datasets/Ionosphere"
    },
    {
        category: "Public Benchmark Datasets",
        name: "Letter",
        dataType: ["tabular"],
        size: "large",
        anomalyRatio: "low",
        description: "Letter recognition dataset (~6.25% anomalies).",
        link: "https://archive.ics.uci.edu/ml/datasets/Letter+Recognition"
    },
    {
        category: "Public Benchmark Datasets",
        name: "Wilt",
        dataType: ["tabular"],
        size: "medium",
        anomalyRatio: "low",
        description: "Remote sensing data for detecting dying trees (~5.41% anomalies).",
        link: "https://archive.ics.uci.edu/ml/datasets/Wilt"
    },
    {
        category: "Public Benchmark Datasets",
        name: "Thyroid",
        dataType: ["tabular"],
        size: "medium",
        anomalyRatio: "medium",
        description: "Thyroid disease detection (~7.41% anomalies).",
        link: "https://archive.ics.uci.edu/ml/datasets/Thyroid+Disease"
    },
    {
        category: "Public Benchmark Datasets",
        name: "Pageblocks",
        dataType: ["tabular"],
        size: "medium",
        anomalyRatio: "low",
        description: "Detects page blocks in documents (~10% anomalies).",
        link: "https://archive.ics.uci.edu/ml/datasets/Page+Blocks+Classification"
    },
    {
        category: "Public Benchmark Datasets",
        name: "Heart",
        dataType: ["tabular"],
        size: "medium",
        anomalyRatio: "high",
        description: "Heart disease detection (~44% anomalies).",
        link: "https://archive.ics.uci.edu/ml/datasets/Heart+Disease"
    },
    {
        category: "Public Benchmark Datasets",
        name: "InternetAds",
        dataType: ["tabular"],
        size: "large",
        anomalyRatio: "medium",
        description: "Advertisement detection dataset (~14.32% anomalies).",
        link: "https://archive.ics.uci.edu/ml/datasets/Internet+Advertisements"
    },
    {
        category: "Public Benchmark Datasets",
        name: "Waveform",
        dataType: ["tabular"],
        size: "large",
        anomalyRatio: "low",
        description: "Waveform classification dataset (~2.73% anomalies).",
        link: "https://archive.ics.uci.edu/ml/datasets/Waveform+Database+Generator+(Version+2)"
    },
    {
        category: "Public Benchmark Datasets",
        name: "Wine",
        dataType: ["tabular"],
        size: "medium",
        anomalyRatio: "low",
        description: "Wine quality dataset (~5.88% anomalies).",
        link: "https://archive.ics.uci.edu/ml/datasets/Wine"
    },
    {
        category: "Public Benchmark Datasets",
        name: "Optdigits",
        dataType: ["tabular"],
        size: "large",
        anomalyRatio: "low",
        description: "Optical digit recognition dataset (~2.88% anomalies).",
        link: "https://archive.ics.uci.edu/ml/datasets/Optical+Recognition+of+Handwritten+Digits"
    },
    {
        category: "Public Benchmark Datasets",
        name: "Glass",
        dataType: ["tabular"],
        size: "medium",
        anomalyRatio: "low",
        description: "Glass classification dataset (~4.49% anomalies).",
        link: "https://archive.ics.uci.edu/ml/datasets/Glass+Identification"
    },
    {
        category: "Public Benchmark Datasets",
        name: "Vehicle",
        dataType: ["tabular"],
        size: "medium",
        anomalyRatio: "high",
        description: "Vehicle classification dataset (~50% anomalies).",
        link: "https://archive.ics.uci.edu/ml/datasets/Statlog+(Vehicle+Silhouettes)"
    },
    {
        category: "Public Benchmark Datasets",
        name: "Vowels",
        dataType: ["tabular"],
        size: "medium",
        anomalyRatio: "low",
        description: "Vowel classification dataset (~3.63% anomalies).",
        link: "https://www.openml.org/search?type=data&sort=runs&id=307&status=active"
    },
    {
        category: "Public Benchmark Datasets",
        name: "Vertebral",
        dataType: ["tabular"],
        size: "medium",
        anomalyRatio: "medium",
        description: "Vertebral column data (~23% anomalies).",
        link: "https://archive.ics.uci.edu/ml/datasets/Vertebral+Column"
    },
    {
        category: "Public Benchmark Datasets",
        name: "Yeast",
        dataType: ["tabular"],
        size: "large",
        anomalyRatio: "medium",
        description: "Yeast cell-cycle regulation (~28.9% anomalies).",
        link: "https://archive.ics.uci.edu/ml/datasets/Yeast"
    },
    {
        category: "Public Benchmark Datasets",
        name: "Spambase",
        dataType: ["tabular"],
        size: "large",
        anomalyRatio: "high",
        description: "Email spam classification (~39.4% anomalies).",
        link: "https://archive.ics.uci.edu/ml/datasets/spambase"
    },
    {
        category: "Public Benchmark Datasets",
        name: "Abalone",
        dataType: ["tabular"],
        size: "medium",
        anomalyRatio: "low",
        description: "Predicts abalone age (~4.29% anomalies).",
        link: "https://archive.ics.uci.edu/ml/datasets/Abalone"
    },
    {
        category: "Public Benchmark Datasets",
        name: "Thyroid-Sick",
        dataType: ["tabular"],
        size: "medium",
        anomalyRatio: "medium",
        description: "Variant of thyroid dataset for detecting sickness (~7.4% anomalies).",
        link: "https://archive.ics.uci.edu/ml/datasets/Thyroid+Disease"
    },
    {
        category: "Public Benchmark Datasets",
        name: "Breast-Cancer",
        dataType: ["tabular"],
        size: "large",
        anomalyRatio: "high",
        description: "Variant of the breast cancer dataset (~33% anomalies).",
        link: "https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)"
    },
    {
        category: "Public Benchmark Datasets",
        name: "Ecoli",
        dataType: ["tabular"],
        size: "medium",
        anomalyRatio: "medium",
        description: "Protein localization dataset (~16% anomalies).",
        link: "https://archive.ics.uci.edu/ml/datasets/Ecoli"
    },

    // 2. Synthetic Datasets
    {
        category: "Synthetic Datasets",
        name: "Synthetic Big Dataset for Anomaly Detection - Kaggle",
        dataType: ["tabular"],
        size: "large",
        anomalyRatio: "high",
        description: "Contains 2M rows for anomaly detection across various income and job scenarios.",
        link: "https://www.kaggle.com/datasets/elouataouiwidad/synthetic-bigdataset-anomalydetection"
    },
    {
        category: "Synthetic Datasets",
        name: "Synthetic Financial Datasets for Fraud Detection - Kaggle",
        dataType: ["tabular"],
        size: "large",
        anomalyRatio: "high",
        description: "Generated by the PaySim mobile money simulator for financial anomaly detection.",
        link: "https://www.kaggle.com/datasets/ealaxi/paysim1"
    },
    {
        category: "Synthetic Datasets",
        name: "GitHub Repository",
        dataType: ["time-series"],
        size: "large",
        anomalyRatio: "varied",
        description: "Provides links to over 250 public time series datasets for anomaly detection.",
        link: "https://github.com/elisejiuqizhang/TS-AD-Datasets"
    },

    // 3. Complex CV/NLP Datasets
    {
        category: "Complex CV/NLP Datasets",
        name: "CIFAR-10",
        dataType: ["image"],
        size: "large",
        anomalyRatio: "low",
        description: "Image classification dataset modified for anomaly detection.",
        link: "https://www.cs.toronto.edu/~kriz/cifar.html"
    },
    {
        category: "Complex CV/NLP Datasets",
        name: "MNIST",
        dataType: ["image"],
        size: "large",
        anomalyRatio: "low",
        description: "Handwritten digit dataset modified for anomaly detection.",
        link: "https://yann.lecun.com/exdb/mnist/"
    },
    {
        category: "Complex CV/NLP Datasets",
        name: "Fashion-MNIST",
        dataType: ["image"],
        size: "large",
        anomalyRatio: "low",
        description: "Fashion articles dataset modified for anomaly detection.",
        link: "https://github.com/zalandoresearch/fashion-mnist"
    },
    {
        category: "Complex CV/NLP Datasets",
        name: "20 Newsgroups",
        dataType: ["text"],
        size: "medium",
        anomalyRatio: "medium",
        description: "Text classification dataset modified by injecting anomalies.",
        link: "http://qwone.com/~jason/20Newsgroups/"
    },
    {
        category: "Complex CV/NLP Datasets",
        name: "Reuters-21578",
        dataType: ["text"],
        size: "large",
        anomalyRatio: "medium",
        description: "Text classification dataset for document classification.",
        link: "https://kdd.ics.uci.edu/databases/reuters21578/reuters21578.html"
    },
    {
        category: "Complex CV/NLP Datasets",
        name: "GTSRB (German Traffic Sign Recognition Benchmark)",
        dataType: ["image"],
        size: "large",
        anomalyRatio: "low",
        description: "Traffic sign image recognition dataset modified for anomaly detection.",
        link: "https://benchmark.ini.rub.de/gtsrb_dataset.html"
    },
    {
        category: "Complex CV/NLP Datasets",
        name: "SVHN (Street View House Numbers)",
        dataType: ["image"],
        size: "large",
        anomalyRatio: "low",
        description: "Street number recognition dataset modified for anomaly detection.",
        link: "http://ufldl.stanford.edu/housenumbers/"
    },

    // 4. Imbalanced Datasets
    {
        category: "Imbalanced Datasets",
        name: "Fraud Detection (Credit Card)",
        dataType: ["tabular"],
        size: "large",
        anomalyRatio: "high",
        description: "Financial transaction dataset for detecting credit card fraud (highly imbalanced).",
        link: "https://www.kaggle.com/mlg-ulb/creditcardfraud"
    },

    // 5. Noisy/Corrupted Datasets
    {
        category: "Noisy/Corrupted Datasets",
        name: "SoftPatch: Unsupervised Anomaly Detection with Noisy Data",
        dataType: ["tabular"],
        size: "medium",
        anomalyRatio: "high",
        description: "A memory-based method to denoise data at the patch level for anomaly detection.",
        link: "https://arxiv.org/html/2403.14233v1"
    },
    {
        category: "Noisy/Corrupted Datasets",
        name: "Robust Anomaly Detection on Unreliable Data",
        dataType: ["tabular"],
        size: "medium",
        anomalyRatio: "high",
        description: "Discusses robust anomaly detection on datasets with noisy labels and unreliable data.",
        link: "https://hal.science/hal-02056558/document"
    }
    
];

function filterResults() {
    const results = document.getElementById('results');
    results.innerHTML = ''; // Clear previous results

    // Filter selections
    const isTabular = document.getElementById("tabular").checked;
    const isTimeSeries = document.getElementById("time-series").checked;
    const isImage = document.getElementById("image").checked;
    const isText = document.getElementById("text").checked;

    const isUnsupervised = document.getElementById("unsupervised").checked;
    const isSemiSupervised = document.getElementById("semi-supervised").checked;
    const isSupervised = document.getElementById("supervised").checked;

    const sizeSelected = document.querySelector('input[name="size"]:checked');
    const anomalySelected = document.querySelector('input[name="anomaly"]:checked');


    // Loop through each item in the data array and apply filters
    data.forEach(item => {
        let show = true;

        // Filter by data type
        if (
            (isTabular && item.dataType.includes("tabular")) ||
            (isTimeSeries && item.dataType.includes("time-series")) ||
            (isImage && item.dataType.includes("image")) ||
            (isText && item.dataType.includes("text"))
        ) {
            show = true;
        } else {
            show = false;
        }

        // Filter by algorithm type (if it's an algorithm)
        if (item.type === "algorithm") {
            if (
                (isUnsupervised && item.algorithmType === "unsupervised") ||
                (isSemiSupervised && item.algorithmType === "semi-supervised") ||
                (isSupervised && item.algorithmType === "supervised")
            ) {
                show = show && true;
            } else {
                show = false;
            }
        }

        // Filter by size
        if (sizeSelected && item.size !== sizeSelected.id) {
            show = false;
        }

        // Filter by anomaly ratio
        if (anomalySelected && item.anomalyRatio !== anomalySelected.id.split("-")[0]) {
            show = false;
        }

        // If all conditions match, display the result
        if (show) {
            const div = document.createElement("div");
            div.classList.add("result");
            div.style.display = "block"; // Make sure result is visible
            div.innerHTML = `<h4>${item.name}</h4><p>${item.description}</p><a href="${item.link}" target="_blank">Learn more</a>`;
            results.appendChild(div);
        }
    });
}
