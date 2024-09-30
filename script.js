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
        name: "XGBoost",
        algorithmType: "supervised",
        dataType: ["tabular"],
        size: "large",
        anomalyRatio: "medium",
        description: "Supervised, gradient boosting machine learning method for classification tasks, including anomaly detection.",
        link: "https://dl.acm.org/doi/10.1145/2939672.2939785"
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
    // More algorithms can be added here...

    // Datasets from datasets.md
    {
        type: "dataset",
        name: "KDDCup99",
        dataType: ["tabular"],
        size: "large",
        anomalyRatio: "medium",
        description: "Network intrusion detection dataset, used to identify abnormal network traffic. (~3.92% anomalies).",
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
    // More datasets can be added here...
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
