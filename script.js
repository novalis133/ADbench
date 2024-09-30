// Full data from both `algorithms.md` and `datasets.md`
// Data for algorithms and datasets is structured below
const data = [
    // Algorithms from algorithms.md
    {
        type: "algorithm",
        name: "Isolation Forest",
        algorithmType: "unsupervised",
        dataType: ["tabular"],
        size: "medium",
        anomalyRatio: "low",
        description: "Unsupervised, tree-based method for detecting anomalies by isolating points in the data space.",
        link: "https://link.springer.com/article/10.1007/s10115-008-0116-6"
    },
    {
        type: "algorithm",
        name: "LOF (Local Outlier Factor)",
        algorithmType: "unsupervised",
        dataType: ["tabular"],
        size: "small",
        anomalyRatio: "low",
        description: "A density-based anomaly detection algorithm that identifies local outliers by comparing the local density of data points.",
        link: "https://dl.acm.org/doi/10.1145/342009.335388"
    },
    {
        type: "algorithm",
        name: "DeepSVDD",
        algorithmType: "unsupervised",
        dataType: ["tabular", "image"],
        size: "large",
        anomalyRatio: "high",
        description: "A deep neural network extension of Support Vector Data Description (SVDD), used to map data into a compact latent space.",
        link: "https://arxiv.org/abs/1802.06822"
    },
    {
        type: "algorithm",
        name: "PCA (Principal Component Analysis)",
        algorithmType: "unsupervised",
        dataType: ["tabular", "image"],
        size: "medium",
        anomalyRatio: "medium",
        description: "Reduces the dimensionality of the data by transforming it into principal components, identifying anomalies that do not conform to the data's structure.",
        link: "https://link.springer.com/article/10.1007/BF02289209"
    },
    {
        type: "algorithm",
        name: "KNN (K-Nearest Neighbors)",
        algorithmType: "unsupervised",
        dataType: ["tabular"],
        size: "medium",
        anomalyRatio: "low",
        description: "An unsupervised anomaly detection algorithm that calculates the distance between data points and their nearest neighbors to detect outliers.",
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
        description: "A simple, non-parametric outlier detection algorithm that detects anomalies using empirical cumulative distribution functions.",
        link: "https://arxiv.org/abs/2012.00390"
    },
    {
        type: "algorithm",
        name: "DAGMM (Deep Autoencoding Gaussian Mixture Model)",
        algorithmType: "unsupervised",
        dataType: ["tabular", "image"],
        size: "large",
        anomalyRatio: "high",
        description: "Combines deep autoencoders and Gaussian Mixture Models (GMMs) to perform anomaly detection by learning compact representations and capturing the underlying data distribution.",
        link: "https://openreview.net/forum?id=BJJLHbb0-"
    },
    {
        type: "algorithm",
        name: "HBOS (Histogram-based Outlier Score)",
        algorithmType: "unsupervised",
        dataType: ["tabular"],
        size: "medium",
        anomalyRatio: "low",
        description: "An unsupervised anomaly detection algorithm that uses histograms to calculate the outlier score of data points based on feature distribution.",
        link: "https://link.springer.com/chapter/10.1007/978-3-642-24477-3_1"
    },
    {
        type: "algorithm",
        name: "MCD (Minimum Covariance Determinant)",
        algorithmType: "unsupervised",
        dataType: ["tabular"],
        size: "small",
        anomalyRatio: "low",
        description: "A robust statistical method that identifies anomalies by finding the minimum covariance determinant in the data distribution.",
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
        description: "A technique for detecting outliers in high-dimensional data by examining the data's subspace structure.",
        link: "https://link.springer.com/article/10.1007/s10115-009-0272-8"
    },
    {
        type: "algorithm",
        name: "RPCA (Robust Principal Component Analysis)",
        algorithmType: "unsupervised",
        dataType: ["tabular", "image"],
        size: "medium",
        anomalyRatio: "low",
        description: "A robust form of PCA that decomposes data into low-rank and sparse matrices, identifying anomalies as sparse components.",
        link: "https://statweb.stanford.edu/~candes/papers/RobustPCA.pdf"
    },
    {
        type: "algorithm",
        name: "DeepSAD (Deep Semi-Supervised Anomaly Detection)",
        algorithmType: "semi-supervised",
        dataType: ["tabular", "image"],
        size: "large",
        anomalyRatio: "high",
        description: "Combines labeled and unlabeled data to learn compact representations of normal instances in the latent space, detecting anomalies based on deviations.",
        link: "https://arxiv.org/abs/2002.00833"
    },
    {
        type: "algorithm",
        name: "REPEN (Representation Learning for Anomaly Detection)",
        algorithmType: "semi-supervised",
        dataType: ["tabular"],
        size: "large",
        anomalyRatio: "medium",
        description: "Learns low-dimensional representations for anomaly detection by minimizing intra-class distances and maximizing inter-class distances.",
        link: "https://www.ijcai.org/proceedings/2017/0221.pdf"
    },
    {
        type: "algorithm",
        name: "GANomaly (Generative Adversarial Network for Anomaly Detection)",
        algorithmType: "semi-supervised",
        dataType: ["image", "tabular"],
        size: "large",
        anomalyRatio: "high",
        description: "A semi-supervised GAN-based model that generates normal data and detects anomalies as points that cannot be well-reconstructed by the generator.",
        link: "https://arxiv.org/abs/1805.06725"
    },
    {
        type: "algorithm",
        name: "DevNet (Deep Anomaly Detection with Deviations)",
        algorithmType: "semi-supervised",
        dataType: ["tabular"],
        size: "large",
        anomalyRatio: "high",
        description: "Detects deviations from normal data patterns by using a neural network that is trained on both labeled and unlabeled data.",
        link: "https://arxiv.org/abs/2002.12718"
    },
    {
        type: "algorithm",
        name: "RDP (Robust Deep PCA)",
        algorithmType: "semi-supervised",
        dataType: ["tabular", "image"],
        size: "large",
        anomalyRatio: "medium",
        description: "Combines deep learning with PCA to create robust representations of data for detecting anomalies in noisy or high-dimensional datasets.",
        link: "https://arxiv.org/abs/1703.08383"
    },
    {
        type: "algorithm",
        name: "SO-GAAL (Single-Objective Generative Adversarial Active Learning)",
        algorithmType: "semi-supervised",
        dataType: ["tabular"],
        size: "large",
        anomalyRatio: "high",
        description: "A GAN-based model that generates synthetic anomalies and uses active learning to detect real anomalies from a small amount of labeled data.",
        link: "https://www.aaai.org/ojs/index.php/AAAI/article/view/4263"
    },
    {
        type: "algorithm",
        name: "XGBoost",
        algorithmType: "supervised",
        dataType: ["tabular"],
        size: "large",
        anomalyRatio: "medium",
        description: "A gradient boosting framework for detecting anomalies in supervised settings, widely used for tabular data.",
        link: "https://dl.acm.org/doi/10.1145/2939672.2939785"
    },
    {
        type: "algorithm",
        name: "LightGBM",
        algorithmType: "supervised",
        dataType: ["tabular"],
        size: "large",
        anomalyRatio: "medium",
        description: "A faster and more efficient version of gradient boosting, often used for large-scale datasets.",
        link: "https://dl.acm.org/doi/10.5555/3294996.3295074"
    },
    {
        type: "algorithm",
        name: "CatBoost",
        algorithmType: "supervised",
        dataType: ["tabular"],
        size: "large",
        anomalyRatio: "medium",
        description: "A gradient boosting method that handles categorical features effectively for anomaly detection tasks.",
        link: "https://arxiv.org/abs/1706.09516"
    },
    {
        type: "algorithm",
        name: "Random Forest",
        algorithmType: "supervised",
        dataType: ["tabular"],
        size: "large",
        anomalyRatio: "medium",
        description: "An ensemble learning method that uses multiple decision trees to improve the accuracy of anomaly detection.",
        link: "https://link.springer.com/article/10.1023/A:1010933404324"
    },
    {
        type: "algorithm",
        name: "MLP (Multilayer Perceptron)",
        algorithmType: "supervised",
        dataType: ["tabular", "image"],
        size: "medium",
        anomalyRatio: "medium",
        description: "A simple feedforward neural network model used for binary classification tasks, including anomaly detection.",
        link: "https://www.nature.com/articles/323533a0"
    },
    {
        type: "algorithm",
        name: "ResNet",
        algorithmType: "supervised",
        dataType: ["image"],
        size: "large",
        anomalyRatio: "medium",
        description: "A deep residual network model used for image classification and anomaly detection tasks, known for its deep architecture.",
        link: "https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf"
    },
    {
        type: "algorithm",
        name: "FTTransformer",
        algorithmType: "supervised",
        dataType: ["tabular"],
        size: "large",
        anomalyRatio: "medium",
        description: "A transformer-based model designed for anomaly detection in high-dimensional tabular data.",
        link: "https://arxiv.org/abs/2106.11959"
    },
    {
        type: "algorithm",
        name: "Naive Bayes",
        algorithmType: "supervised",
        dataType: ["tabular"],
        size: "small",
        anomalyRatio: "low",
        description: "A probabilistic classifier that uses Bayes' theorem with strong (naive) independence assumptions between the features.",
        link: "https://dl.acm.org/doi/10.5555/599619.599635"
    },
    {
        type: "algorithm",
        name: "Logistic Regression",
        algorithmType: "supervised",
        dataType: ["tabular"],
        size: "small",
        anomalyRatio: "medium",
        description: "A binary classification model that estimates the probability of a given data point being anomalous or normal based on input features.",
        link: "https://www.jstor.org/stable/2237932"
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

// Function to filter results based on user selection
function filterResults() {
    const isTabular = document.getElementById("tabular").checked;
    const isTimeSeries = document.getElementById("time-series").checked;
    const isImage = document.getElementById("image").checked;
    const isText = document.getElementById("text").checked;

    const isUnsupervised = document.getElementById("unsupervised").checked;
    const isSemiSupervised = document.getElementById("semi-supervised").checked;
    const isSupervised = document.getElementById("supervised").checked;

    const sizeSelected = document.querySelector('input[name="size"]:checked');
    const anomalySelected = document.querySelector('input[name="anomaly"]:checked');

    const results = document.getElementById("results");
    results.innerHTML = "";

    data.forEach(item => {
        let show = false;

        // Filter based on data type
        if (
            (isTabular && item.dataType.includes("tabular")) ||
            (isTimeSeries && item.dataType.includes("time-series")) ||
            (isImage && item.dataType.includes("image")) ||
            (isText && item.dataType.includes("text"))
        ) {
            show = true;
        }

        // Filter based on algorithm type (if it's an algorithm)
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

        // Filter based on dataset size
        if (sizeSelected && item.size !== sizeSelected.id) {
            show = false;
        }

        // Filter based on anomaly ratio
        if (anomalySelected && item.anomalyRatio !== anomalySelected.id.split("-")[0]) {
            show = false;
        }

        if (show) {
            const div = document.createElement("div");
            div.classList.add("result");
            div.innerHTML = `<h4>${item.name}</h4><p>${item.description}</p><a href="${item.link}" target="_blank">Learn more</a>`;
            results.appendChild(div);
        }
    });
}
