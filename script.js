// Full data from both `algorithms.md` and `datasets.md`
// Data for algorithms and datasets is structured below
const data = [
    {
        type: "algorithm",
        name: "Isolation Forest",
        algorithmType: "unsupervised",
        dataType: ["tabular"],
        size: "medium",
        anomalyRatio: "low",
        description: "Tree-based method for detecting anomalies by isolating points in the data space.",
        link: "https://link.springer.com/article/10.1007/s10115-008-0116-6"
    },
    {
        type: "algorithm",
        name: "LOF (Local Outlier Factor)",
        algorithmType: "unsupervised",
        dataType: ["tabular"],
        size: "small",
        anomalyRatio: "low",
        description: "Density-based algorithm identifying local outliers by comparing the local density of data points.",
        link: "https://dl.acm.org/doi/10.1145/342009.335388"
    },
    {
        type: "algorithm",
        name: "DeepSVDD",
        algorithmType: "unsupervised",
        dataType: ["tabular", "image"],
        size: "large",
        anomalyRatio: "high",
        description: "Deep neural network extension of SVDD for mapping data into a compact latent space.",
        link: "https://arxiv.org/abs/1802.06822"
    },
    {
        type: "algorithm",
        name: "PCA (Principal Component Analysis)",
        algorithmType: "unsupervised",
        dataType: ["tabular", "image"],
        size: "medium",
        anomalyRatio: "medium",
        description: "Reduces the dimensionality of the data by transforming it into principal components.",
        link: "https://link.springer.com/article/10.1007/BF02289209"
    },
    {
        type: "algorithm",
        name: "KNN (K-Nearest Neighbors)",
        algorithmType: "unsupervised",
        dataType: ["tabular"],
        size: "medium",
        anomalyRatio: "low",
        description: "Calculates the distance between data points and their nearest neighbors to detect outliers.",
        link: "https://www.sciencedirect.com/science/article/abs/pii/S0957417405002199"
    },
    {
        type: "algorithm",
        name: "COPOD (Copula-Based Outlier Detection)",
        algorithmType: "unsupervised",
        dataType: ["tabular"],
        size: "medium",
        anomalyRatio: "medium",
        description: "Uses copula functions to model dependencies and detect outliers by calculating empirical CDF.",
        link: "https://arxiv.org/abs/2009.09463"
    },
    {
        type: "algorithm",
        name: "ECOD (Empirical Cumulative Distribution Outlier Detection)",
        algorithmType: "unsupervised",
        dataType: ["tabular"],
        size: "large",
        anomalyRatio: "low",
        description: "Non-parametric outlier detection algorithm based on empirical cumulative distribution functions.",
        link: "https://arxiv.org/abs/2012.00390"
    },
    {
        type: "algorithm",
        name: "DAGMM (Deep Autoencoding Gaussian Mixture Model)",
        algorithmType: "unsupervised",
        dataType: ["tabular", "image"],
        size: "large",
        anomalyRatio: "high",
        description: "Combines deep autoencoders and Gaussian Mixture Models to detect anomalies.",
        link: "https://openreview.net/forum?id=BJJLHbb0-"
    },
    {
        type: "algorithm",
        name: "HBOS (Histogram-based Outlier Score)",
        algorithmType: "unsupervised",
        dataType: ["tabular"],
        size: "medium",
        anomalyRatio: "low",
        description: "Uses histograms to calculate outlier score of data points based on feature distribution.",
        link: "https://link.springer.com/chapter/10.1007/978-3-642-24477-3_1"
    },
    {
        type: "algorithm",
        name: "MCD (Minimum Covariance Determinant)",
        algorithmType: "unsupervised",
        dataType: ["tabular"],
        size: "small",
        anomalyRatio: "low",
        description: "Identifies anomalies by finding the minimum covariance determinant in the data distribution.",
        link: "https://link.springer.com/article/10.1007/BF01908701"
    },
    {
        type: "algorithm",
        name: "COF (Connectivity-Based Outlier Factor)",
        algorithmType: "unsupervised",
        dataType: ["tabular"],
        size: "medium",
        anomalyRatio: "low",
        description: "Measures the degree of outlierness of an object based on the relative neighborhood connectivity.",
        link: "https://dl.acm.org/doi/10.1145/775047.775053"
    },
    {
        type: "algorithm",
        name: "SOD (Subspace Outlier Detection)",
        algorithmType: "unsupervised",
        dataType: ["tabular"],
        size: "medium",
        anomalyRatio: "medium",
        description: "Detects outliers in high-dimensional data by examining the subspace structure.",
        link: "https://link.springer.com/article/10.1007/s10115-009-0272-8"
    },
    {
        type: "algorithm",
        name: "RPCA (Robust Principal Component Analysis)",
        algorithmType: "unsupervised",
        dataType: ["tabular", "image"],
        size: "medium",
        anomalyRatio: "low",
        description: "Decomposes data into low-rank and sparse matrices to detect anomalies as sparse components.",
        link: "https://statweb.stanford.edu/~candes/papers/RobustPCA.pdf"
    },
    {
        type: "algorithm",
        name: "DeepSAD (Deep Semi-Supervised Anomaly Detection)",
        algorithmType: "semi-supervised",
        dataType: ["tabular", "image"],
        size: "large",
        anomalyRatio: "high",
        description: "Combines labeled and unlabeled data to learn compact representations of normal instances.",
        link: "https://arxiv.org/abs/2002.00833"
    },
    {
        type: "algorithm",
        name: "REPEN (Representation Learning for Anomaly Detection)",
        algorithmType: "semi-supervised",
        dataType: ["tabular"],
        size: "large",
        anomalyRatio: "medium",
        description: "Learns low-dimensional representations for anomaly detection by minimizing intra-class distances.",
        link: "https://www.ijcai.org/proceedings/2017/0221.pdf"
    },
    {
        type: "algorithm",
        name: "GANomaly (Generative Adversarial Network for Anomaly Detection)",
        algorithmType: "semi-supervised",
        dataType: ["image", "tabular"],
        size: "large",
        anomalyRatio: "high",
        description: "A GAN-based model that generates normal data and detects anomalies by reconstruction errors.",
        link: "https://arxiv.org/abs/1805.06725"
    },
    {
        type: "algorithm",
        name: "DevNet (Deep Anomaly Detection with Deviations)",
        algorithmType: "semi-supervised",
        dataType: ["tabular"],
        size: "large",
        anomalyRatio: "high",
        description: "Detects deviations from normal data patterns by using a neural network trained on labeled data.",
        link: "https://arxiv.org/abs/2002.12718"
    },
    {
        type: "algorithm",
        name: "XGBoost",
        algorithmType: "supervised",
        dataType: ["tabular"],
        size: "large",
        anomalyRatio: "medium",
        description: "Gradient boosting framework widely used for tabular data in supervised settings.",
        link: "https://dl.acm.org/doi/10.1145/2939672.2939785"
    },
    {
        type: "algorithm",
        name: "LightGBM",
        algorithmType: "supervised",
        dataType: ["tabular"],
        size: "large",
        anomalyRatio: "medium",
        description: "Efficient version of gradient boosting, often used for large-scale datasets.",
        link: "https://dl.acm.org/doi/10.5555/3294996.3295074"
    },
    {
        type: "algorithm",
        name: "CatBoost",
        algorithmType: "supervised",
        dataType: ["tabular"],
        size: "large",
        anomalyRatio: "medium",
        description: "Gradient boosting method handling categorical features effectively for anomaly detection.",
        link: "https://arxiv.org/abs/1706.09516"
    },
    {
        type: "algorithm",
        name: "Random Forest",
        algorithmType: "supervised",
        dataType: ["tabular"],
        size: "large",
        anomalyRatio: "medium",
        description: "Ensemble learning method using multiple decision trees for anomaly detection.",
        link: "https://link.springer.com/article/10.1023/A:1010933404324"
    },
    {
        type: "algorithm",
        name: "MLP (Multilayer Perceptron)",
        algorithmType: "supervised",
        dataType: ["tabular", "image"],
        size: "medium",
        anomalyRatio: "medium",
        description: "Feedforward neural network model used for binary classification and anomaly detection.",
        link: "https://www.nature.com/articles/323533a0"
    },
    {
        type: "algorithm",
        name: "ResNet",
        algorithmType: "supervised",
        dataType: ["image"],
        size: "large",
        anomalyRatio: "medium",
        description: "Deep residual network used for image classification and anomaly detection tasks.",
        link: "https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf"
    },
    {
        type: "algorithm",
        name: "FTTransformer",
        algorithmType: "supervised",
        dataType: ["tabular"],
        size: "large",
        anomalyRatio: "medium",
        description: "Transformer-based model designed for anomaly detection in high-dimensional tabular data.",
        link: "https://arxiv.org/abs/2106.11959"
    },
    {
        type: "algorithm",
        name: "Naive Bayes",
        algorithmType: "supervised",
        dataType: ["tabular"],
        size: "small",
        anomalyRatio: "low",
        description: "A probabilistic classifier using Bayes' theorem with strong independence assumptions between the features.",
        link: "https://dl.acm.org/doi/10.5555/599619.599635"
    },
    {
        type: "algorithm",
        name: "Logistic Regression",
        algorithmType: "supervised",
        dataType: ["tabular"],
        size: "small",
        anomalyRatio: "medium",
        description: "Binary classification model estimating the probability of a data point being anomalous based on input features.",
        link: "https://www.jstor.org/stable/2237932"
    },
    {
        type: "algorithm",
        name: "SO-GAAL (Single-Objective Generative Adversarial Active Learning)",
        algorithmType: "semi-supervised",
        dataType: ["tabular"],
        size: "large",
        anomalyRatio: "high",
        description: "A GAN-based model generating synthetic anomalies and using active learning to detect real anomalies.",
        link: "https://www.aaai.org/ojs/index.php/AAAI/article/view/4263"
    },
    {
        type: "algorithm",
        name: "RDP (Robust Deep PCA)",
        algorithmType: "semi-supervised",
        dataType: ["tabular", "image"],
        size: "large",
        anomalyRatio: "medium",
        description: "Deep learning-based PCA model designed for robust anomaly detection in high-dimensional data.",
        link: "https://arxiv.org/abs/1703.08383"
    },
    {
        type: "algorithm",
        name: "HBOS (Histogram-based Outlier Score)",
        algorithmType: "unsupervised",
        dataType: ["tabular"],
        size: "medium",
        anomalyRatio: "medium",
        description: "Anomaly detection method that assumes independence between features and calculates outlier scores using histograms.",
        link: "https://ieeexplore.ieee.org/document/6085272"
    },
    {
        type: "dataset",
        name: "Abalone",
        dataType: ["tabular"],
        size: "medium",
        anomalyRatio: "low",
        description: "Predicts the age of abalone from physical measurements (~4.29% anomalies).",
        link: "https://archive.ics.uci.edu/ml/datasets/Abalone"
    },
    {
        type: "dataset",
        name: "Glass Identification",
        dataType: ["tabular"],
        size: "medium",
        anomalyRatio: "low",
        description: "Dataset used for glass type classification (~4.49% anomalies).",
        link: "https://archive.ics.uci.edu/ml/datasets/Glass+Identification"
    },
    {
        type: "dataset",
        name: "Vehicle Silhouettes",
        dataType: ["tabular"],
        size: "medium",
        anomalyRatio: "medium",
        description: "Vehicle silhouette classification dataset (~50% anomalies).",
        link: "https://archive.ics.uci.edu/ml/datasets/Statlog+(Vehicle+Silhouettes)"
    },
    {
        type: "dataset",
        name: "Yeast",
        dataType: ["tabular"],
        size: "large",
        anomalyRatio: "medium",
        description: "Yeast gene regulation dataset for anomaly detection (~28.9% anomalies).",
        link: "https://archive.ics.uci.edu/ml/datasets/Yeast"
    },
    {
        type: "dataset",
        name: "Vertebral Column",
        dataType: ["tabular"],
        size: "medium",
        anomalyRatio: "medium",
        description: "Vertebral column dataset for classifying orthopedic conditions (~23% anomalies).",
        link: "https://archive.ics.uci.edu/ml/datasets/Vertebral+Column"
    },
    {
        type: "dataset",
        name: "Forest Cover Type",
        dataType: ["tabular"],
        size: "large",
        anomalyRatio: "medium",
        description: "Predicts forest cover type from cartographic variables (~0.96% anomalies).",
        link: "https://archive.ics.uci.edu/ml/datasets/covertype"
    },
    {
        type: "dataset",
        name: "Cardiotocography",
        dataType: ["tabular"],
        size: "medium",
        anomalyRatio: "medium",
        description: "Fetal health classification from cardiotocograms (~22% anomalies).",
        link: "https://archive.ics.uci.edu/ml/datasets/Cardiotocography"
    },
    {
        type: "dataset",
        name: "Steel Plates Faults",
        dataType: ["tabular"],
        size: "large",
        anomalyRatio: "medium",
        description: "Dataset for classifying faults in steel plates (~6.45% anomalies).",
        link: "https://archive.ics.uci.edu/ml/datasets/Steel+Plates+Faults"
    },
    
    // Datasets from datasets.md
    {
        type: "dataset",
        name: "KDDCup99",
        dataType: ["tabular"],
        size: "large",
        anomalyRatio: "medium",
        description: "Network intrusion detection dataset used to identify abnormal network traffic (~3.92% anomalies).",
        link: "https://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html"
    },
    {
        type: "dataset",
        name: "MNIST",
        dataType: ["image"],
        size: "large",
        anomalyRatio: "low",
        description: "A large dataset of handwritten digits, used for image-based anomaly detection (~1% anomalies).",
        link: "https://yann.lecun.com/exdb/mnist/"
    },
    {
        type: "dataset",
        name: "Arrhythmia",
        dataType: ["tabular"],
        size: "medium",
        anomalyRatio: "medium",
        description: "Healthcare dataset for detecting cardiac arrhythmias (~15% anomalies).",
        link: "https://archive.ics.uci.edu/ml/datasets/arrhythmia"
    },
    {
        type: "dataset",
        name: "Annthyroid",
        dataType: ["tabular"],
        size: "medium",
        anomalyRatio: "high",
        description: "Medical dataset for thyroid disease detection (~21% anomalies).",
        link: "https://archive.ics.uci.edu/ml/datasets/Thyroid+Disease"
    },
    {
        type: "dataset",
        name: "Spambase",
        dataType: ["tabular"],
        size: "large",
        anomalyRatio: "high",
        description: "Email spam detection dataset, used for classifying spam emails (~39.4% anomalies).",
        link: "https://archive.ics.uci.edu/ml/datasets/spambase"
    },
    {
        type: "dataset",
        name: "Satellite",
        dataType: ["tabular"],
        size: "large",
        anomalyRatio: "medium",
        description: "Remote sensing dataset for satellite image classification (~31% anomalies).",
        link: "https://archive.ics.uci.edu/ml/datasets/Statlog+(Landsat+Satellite)"
    },
    {
        type: "dataset",
        name: "Shuttle",
        dataType: ["tabular"],
        size: "large",
        anomalyRatio: "low",
        description: "NASA dataset for shuttle failure classification (~7.15% anomalies).",
        link: "https://archive.ics.uci.edu/ml/datasets/Statlog+(Shuttle)"
    },
    {
        type: "dataset",
        name: "Musk",
        dataType: ["tabular"],
        size: "medium",
        anomalyRatio: "medium",
        description: "Dataset used to detect whether a molecule is 'musky' (~3.17% anomalies).",
        link: "https://archive.ics.uci.edu/ml/datasets/Musk+(Version+2)"
    },
    {
        type: "dataset",
        name: "Mammography",
        dataType: ["tabular"],
        size: "large",
        anomalyRatio: "low",
        description: "Breast cancer detection dataset (~2.32% anomalies).",
        link: "https://archive.ics.uci.edu/ml/datasets/Mammographic+Mass"
    },
    {
        type: "dataset",
        name: "Breastw",
        dataType: ["tabular"],
        size: "large",
        anomalyRatio: "high",
        description: "Wisconsin breast cancer dataset (~34.47% anomalies).",
        link: "https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)"
    },
    {
        type: "dataset",
        name: "Pima",
        dataType: ["tabular"],
        size: "medium",
        anomalyRatio: "medium",
        description: "Diabetes detection dataset (~35% anomalies).",
        link: "https://archive.ics.uci.edu/ml/datasets/Pima+Indians+Diabetes"
    },
    {
        type: "dataset",
        name: "Ionosphere",
        dataType: ["tabular"],
        size: "medium",
        anomalyRatio: "medium",
        description: "Radar data for classifying ionosphere structure (~35% anomalies).",
        link: "https://archive.ics.uci.edu/ml/datasets/Ionosphere"
    },
    {
        type: "dataset",
        name: "Letter",
        dataType: ["tabular"],
        size: "large",
        anomalyRatio: "low",
        description: "Letter recognition dataset (~6.25% anomalies).",
        link: "https://archive.ics.uci.edu/ml/datasets/Letter+Recognition"
    },
    {
        type: "dataset",
        name: "Wilt",
        dataType: ["tabular"],
        size: "medium",
        anomalyRatio: "low",
        description: "Remote sensing data for detecting dying trees (~5.41% anomalies).",
        link: "https://archive.ics.uci.edu/ml/datasets/Wilt"
    },
    {
        type: "dataset",
        name: "Pageblocks",
        dataType: ["tabular"],
        size: "medium",
        anomalyRatio: "low",
        description: "Detects page blocks in documents (~10% anomalies).",
        link: "https://archive.ics.uci.edu/ml/datasets/Page+Blocks+Classification"
    },
    {
        type: "dataset",
        name: "Heart",
        dataType: ["tabular"],
        size: "medium",
        anomalyRatio: "high",
        description: "Heart disease detection dataset (~44% anomalies).",
        link: "https://archive.ics.uci.edu/ml/datasets/Heart+Disease"
    },
    {
        type: "dataset",
        name: "InternetAds",
        dataType: ["tabular"],
        size: "large",
        anomalyRatio: "medium",
        description: "Advertisement detection dataset (~14.32% anomalies).",
        link: "https://archive.ics.uci.edu/ml/datasets/Internet+Advertisements"
    },
    {
        type: "dataset",
        name: "Penbased",
        dataType: ["tabular"],
        size: "large",
        anomalyRatio: "low",
        description: "Handwritten pen digit dataset (~6.59% anomalies).",
        link: "https://archive.ics.uci.edu/ml/datasets/Pen-Based+Recognition+of+Handwritten+Digits"
    },
    {
        type: "dataset",
        name: "Waveform",
        dataType: ["tabular"],
        size: "large",
        anomalyRatio: "low",
        description: "Waveform classification dataset (~2.73% anomalies).",
        link: "https://archive.ics.uci.edu/ml/datasets/Waveform+Database+Generator+(Version+2)"
    },
    {
        type: "dataset",
        name: "Wine",
        dataType: ["tabular"],
        size: "medium",
        anomalyRatio: "low",
        description: "Wine quality dataset (~5.88% anomalies).",
        link: "https://archive.ics.uci.edu/ml/datasets/Wine"
    },
    {
        type: "dataset",
        name: "Optdigits",
        dataType: ["tabular"],
        size: "large",
        anomalyRatio: "low",
        description: "Optical digit recognition dataset (~2.88% anomalies).",
        link: "https://archive.ics.uci.edu/ml/datasets/Optical+Recognition+of+Handwritten+Digits"
    },
    {
        type: "dataset",
        name: "Glass",
        dataType: ["tabular"],
        size: "medium",
        anomalyRatio: "low",
        description: "Glass classification dataset (~4.49% anomalies).",
        link: "https://archive.ics.uci.edu/ml/datasets/Glass+Identification"
    },
    {
        type: "dataset",
        name: "Vehicle",
        dataType: ["tabular"],
        size: "medium",
        anomalyRatio: "medium",
        description: "Vehicle classification dataset (~50% anomalies).",
        link: "https://archive.ics.uci.edu/ml/datasets/Statlog+(Vehicle+Silhouettes)"
    },
    {
        type: "dataset",
        name: "Vowels",
        dataType: ["tabular"],
        size: "medium",
        anomalyRatio: "low",
        description: "Vowel classification dataset (~3.63% anomalies).",
        link: "https://archive.ics.uci.edu/ml/datasets/Connectionist+Bench+%28Vowel+Recognition+-+Deterding+Data%29"
    },
    {
        type: "dataset",
        name: "Vertebral",
        dataType: ["tabular"],
        size: "medium",
        anomalyRatio: "medium",
        description: "Vertebral column dataset (~23% anomalies).",
        link: "https://archive.ics.uci.edu/ml/datasets/Vertebral+Column"
    },
    {
        type: "dataset",
        name: "Yeast",
        dataType: ["tabular"],
        size: "large",
        anomalyRatio: "medium",
        description: "Yeast cell-cycle regulation dataset (~28.9% anomalies).",
        link: "https://archive.ics.uci.edu/ml/datasets/Yeast"
    },
    {
        type: "dataset",
        name: "Abalone",
        dataType: ["tabular"],
        size: "medium",
        anomalyRatio: "low",
        description: "Predicts abalone age (~4.29% anomalies).",
        link: "https://archive.ics.uci.edu/ml/datasets/Abalone"
    }
    // Add more datasets if needed...
];

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
