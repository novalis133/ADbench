// Sample data representing algorithms and datasets from algorithms.md and datasets.md
const data = [
    {
        type: "algorithm",
        name: "Isolation Forest",
        algorithmType: "unsupervised",
        dataType: ["tabular"],
        size: "medium",
        anomalyRatio: "low",
        description: "Unsupervised, Tree-based anomaly detection",
        link: "https://link.springer.com/article/10.1007/s10115-008-0116-6"
    },
    {
        type: "algorithm",
        name: "XGBoost",
        algorithmType: "supervised",
        dataType: ["tabular"],
        size: "large",
        anomalyRatio: "medium",
        description: "Supervised, Gradient boosting method",
        link: "https://dl.acm.org/doi/10.1145/2939672.2939785"
    },
    {
        type: "dataset",
        name: "KDDCup99",
        dataType: ["tabular"],
        size: "large",
        anomalyRatio: "medium",
        description: "Network intrusion detection dataset",
        link: "https://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html"
    },
    {
        type: "dataset",
        name: "MNIST",
        dataType: ["image"],
        size: "large",
        anomalyRatio: "low",
        description: "Handwritten digits dataset",
        link: "https://yann.lecun.com/exdb/mnist/"
    },
    // Add more algorithms and datasets here...
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