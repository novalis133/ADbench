# Algorithm and Dataset Selector

![Algorithm Selector Banner](https://source.unsplash.com/800x200/?machine-learning,anomaly-detection)  
*Interactive web tool for selecting anomaly detection algorithms and datasets based on user-defined criteria.*

This repository contains a web-based application designed to help researchers and practitioners select the best anomaly detection algorithms and datasets based on criteria such as data type, algorithm type, dataset size, and anomaly ratio. Built with HTML, JavaScript, and Tailwind CSS, the tool filters a curated list of algorithms and datasets, providing detailed descriptions, use cases, and references. As a Senior AI & Machine Learning Engineer, this project demonstrates my expertise in creating user-friendly tools for AI-driven decision-making, complementing my work in neuromorphic computing and reinforcement learning.

[![HTML](https://img.shields.io/badge/HTML-5-E34F26?logo=html5&logoColor=white)](https://developer.mozilla.org/en-US/docs/Web/HTML) 
[![JavaScript](https://img.shields.io/badge/JavaScript-ES6+-F7DF1E?logo=javascript&logoColor=black)](https://developer.mozilla.org/en-US/docs/Web/JavaScript) 
[![Tailwind CSS](https://img.shields.io/badge/Tailwind_CSS-3.4+-38B2AC?logo=tailwind-css&logoColor=white)](https://tailwindcss.com/) 
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📚 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Contributing](#contributing)
- [Contact](#contact)
- [Acknowledgments](#acknowledgments)

---

## 📖 Overview

The Algorithm and Dataset Selector is a web application that allows users to filter anomaly detection algorithms and datasets by selecting criteria such as:
- **Data Type**: Tabular, Time-Series, Image, Text
- **Algorithm Type**: Unsupervised, Semi-supervised, Supervised
- **Dataset Size**: Small, Medium, Large
- **Anomaly Ratio**: Low (<5%), Medium (5-15%), High (>15%)

The tool dynamically displays matching algorithms and datasets with their descriptions, use cases, and links to references. It leverages a curated dataset of 30+ algorithms (e.g., Isolation Forest, XGBoost, DeepSAD) and 40+ datasets (e.g., KDDCup99, MNIST, Fraud Detection), making it a valuable resource for AI/ML practitioners working on anomaly detection tasks.

This project showcases my ability to integrate front-end development with AI/ML domain knowledge, providing a practical tool for the community.

---

## ✨ Features

| Feature | Description | Use Case |
|---------|-------------|----------|
| **Dynamic Filtering** | Filter algorithms and datasets based on multiple criteria. | Quickly find relevant resources |
| **Responsive UI** | Built with Tailwind CSS for a modern, mobile-friendly design. | Accessible on all devices |
| **Accessibility** | Includes ARIA attributes and keyboard navigation. | Inclusive user experience |
| **Debounced Updates** | Prevents excessive filter calls for smooth performance. | Efficient user interaction |
| **Detailed Outputs** | Displays algorithm use cases, citations, and dataset details. | Comprehensive decision support |
| **Error Handling** | Shows loading states and no-results messages. | Robust user feedback |

---

## 🚀 Getting Started

### Prerequisites
- A modern web browser (e.g., Chrome, Firefox, Edge)
- Git (optional, for cloning the repository)
- A local server (optional, for development, e.g., `live-server` or Python’s `http.server`)

### Setup
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Novalis133/Medium.git
   cd Medium/algorithm_dataset_selector
   ```

2. **Open the Application**:
   - Option 1: Open `index.html` directly in a browser.
   - Option 2: Serve locally for development:
     ```bash
     # Using Python
     python -m http.server 8000
     ```
     Then navigate to `http://localhost:8000`.

3. **Explore the Tool**:
   - Select filters (e.g., "Tabular" and "Supervised") to see matching algorithms and datasets.
   - Click "Learn more" links for detailed references.

**Troubleshooting**:
- **Blank Results**: Ensure at least one filter is selected.
- **Styling Issues**: Verify the Tailwind CSS CDN (`https://cdn.tailwindcss.com`) is accessible.
- **JavaScript Errors**: Check the console for errors and ensure `script.js` is in the same directory as `index.html`.

---

## 📂 Project Structure

```
Medium/algorithm_dataset_selector/
├── index.html                    # Main HTML file with UI and Tailwind CSS
├── script.js                     # JavaScript for filtering and rendering results
├── LICENSE                       # MIT License
└── README.md                     # Project documentation
```

---

## 🖥️ Usage

1. **Open the Application**: Load `index.html` in a browser.
2. **Select Filters**:
   - Choose data types (e.g., Tabular, Image).
   - Select algorithm types (e.g., Unsupervised, Supervised).
   - Pick a dataset size and anomaly ratio.
3. **View Results**:
   - The "Algorithm Results" section displays matching algorithms with names, descriptions, use cases, citations, and links.
   - The "Dataset Results" section shows datasets with names, descriptions, data types, sizes, anomaly ratios, and links.
4. **Interact**:
   - Results update dynamically with a 300ms debounce for smooth performance.
   - If no results match, a "No results" message appears.
   - A loading state is shown during filter processing.

**Example**:
- Select "Tabular" and "Supervised" to see algorithms like XGBoost and datasets like KDDCup99.
- Select "Image" and "Low" anomaly ratio to find datasets like MNIST or CIFAR-10.

---

## 🤝 Contributing

Contributions to enhance the selector’s functionality, dataset coverage, or UI are welcome! Follow these steps:
1. **Fork the Repository**:
   ```bash
   git fork https://github.com/Novalis133/Medium.git
   ```
2. **Create a Feature Branch**:
   ```bash
   cd Medium/algorithm_dataset_selector
   git checkout -b feature/add-new-algorithm
   ```
3. **Update Code**:
   - Add new algorithms or datasets to `script.js`.
   - Enhance UI in `index.html` or add custom CSS.
   - Ensure compatibility with existing filters.
4. **Commit Changes**:
   Use [Conventional Commits](https://www.conventionalcommits.org/):
   ```bash
   git commit -m "feat: add new anomaly detection algorithm"
   ```
5. **Test Changes**:
   - Verify filters work correctly in a browser.
   - Check console for JavaScript errors.
   - Ensure Tailwind classes render properly.
6. **Submit a Pull Request**:
   ```bash
   git push origin feature/add-new-algorithm
   ```
   Open a PR with a detailed description.

**Guidelines**:
- Follow the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/).
- Test thoroughly with various filter combinations.
- Update `README.md` for new features or datasets.

---

## 📫 Contact

- **Email**: osama1339669@gmail.com
- **LinkedIn**: [Osama](https://www.linkedin.com/in/osamat339669/)
- **GitHub Issues**: [Issues Page](https://github.com/Novalis133/Medium/issues)
- **Medium Blog**: [Osama’s Medium](https://medium.com/@osama1339669)

---

## 🙏 Acknowledgments

- [Tailwind CSS](https://tailwindcss.com/) for responsive styling.
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/) for dataset sources.
- [Kaggle](https://www.kaggle.com/) for synthetic and fraud detection datasets.
- Academic references for algorithms (e.g., Liu et al., Chen & Guestrin).