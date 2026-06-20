# Customer Segmentation using Unsupervised Learning

A complete end-to-end Machine Learning project that segments mall customers into distinct groups using **K-Means Clustering**, with dimensionality reduction via **PCA**, and translates each cluster into an actionable **business persona**.

---

##  Project Overview

Understanding customer behavior is critical for targeted marketing. This project analyzes customer demographic and spending data to automatically discover natural customer segments — without any predefined labels — using unsupervised learning techniques.

The project is delivered in two forms:
- A **Jupyter Notebook** for step-by-step analysis and learning
- An **interactive Streamlit dashboard** for live exploration

---

##  Project Structure

```
Customer_Segmentation/
│
├── Customer_Segmentation.ipynb   # Full end-to-end analysis notebook
├── app.py                        # Interactive Streamlit dashboard
├── Mall_Customers.csv            # Dataset
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation
```

---

## Dataset

**Mall Customers Dataset** — 200 customer records with the following columns:

| Column | Description |
|---|---|
| `CustomerID` | Unique customer identifier |
| `Gender` | Male / Female |
| `Age` | Customer age |
| `Annual Income (k$)` | Yearly income in thousands of dollars |
| `Spending Score (1-100)` | Score assigned based on spending behavior |

---

##  Machine Learning Pipeline

1. **Data Loading & Exploration** — shape, types, summary statistics, missing values
2. **Data Cleaning** — dropped non-predictive `CustomerID` column
3. **Categorical Encoding** — `Gender` encoded using `LabelEncoder`
4. **Feature Scaling** — standardized all features with `StandardScaler`
5. **Dimensionality Reduction** — `PCA` reduces features to 2 principal components for visualization
6. **Optimal K Selection** — Elbow Method (WCSS vs K) to choose the best number of clusters
7. **Clustering** — `K-Means` applied with the optimal K
8. **Visualization** — 2D (Income vs Spending), PCA view, and 3D (Age, Income, Spending) cluster plots
9. **Business Interpretation** — each cluster mapped to a real-world customer persona

---

## Customer Personas

| Persona | Profile | Recommended Strategy |
|---|---|---|
| **Premium Target Customers** | High income, high spending | Loyalty programs, premium products, personalized offers |
| **Cautious High Earners** | High income, low spending | Value-driven promotions to unlock spending |
| **Impulsive Budget Spenders** | Low income, high spending | Discount bundles, flexible payment options |
| **Budget-Conscious Customers** | Low income, low spending | Low-cost essentials, clearance deals |
| **Average Customers** | Mid income, mid spending | Cross-selling, seasonal campaigns |

---

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.9+ |
| Data Handling | pandas, numpy |
| Machine Learning | scikit-learn (KMeans, PCA, StandardScaler, LabelEncoder) |
| Visualization | matplotlib, plotly |
| Dashboard | Streamlit |
| Notebook | Jupyter |

---

##  Installation

### 1. Clone or download this project
```bash
git clone <your-repo-url>
cd Customer_Segmentation
```

### 2. (Recommended) Create a virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

##  Usage

### Run the Jupyter Notebook
Keep `Mall_Customers.csv` in the same folder, then open:
```bash
jupyter notebook Customer_Segmentation.ipynb
```

### Run the Interactive Dashboard
```bash
streamlit run app.py
```
Then open **http://localhost:8501** in your browser.

**Dashboard capabilities:**
- Upload your own customer CSV or use the default dataset
- Adjust the number of clusters (K) live via slider
- View Elbow Method chart for optimal K
- Explore interactive 2D, PCA, and 3D cluster visualizations
- View auto-generated cluster profile & persona table
- Download the segmented dataset with cluster labels as CSV

---

##  Troubleshooting (Windows)

If `python` or `pip` is not recognized in PowerShell:

1. **Disable the fake Microsoft Store alias:**
   Settings → Apps → Advanced app settings → App execution aliases → turn OFF `python.exe` / `python3.exe`

2. **Use the full path** if Python is installed but not on PATH (e.g., Anaconda):
   ```powershell
   & "C:\ProgramData\Anaconda3\python.exe" -m pip install -r requirements.txt
   & "C:\ProgramData\Anaconda3\python.exe" -m streamlit run app.py
   ```

3. **Add Python permanently to PATH:**
   Edit environment variables for your account → Path → New → add your Python folder and its `Scripts` subfolder → restart terminal.

---

## Future Improvements

- Add Silhouette Score / Davies-Bouldin Index for cluster validation
- Compare K-Means with Hierarchical Clustering and DBSCAN
- Deploy dashboard to Streamlit Cloud / Render for public access
- Add automated cluster-to-persona labeling using business rules engine

---

##  Author

Created as a Data Science portfolio project demonstrating unsupervised learning, dimensionality reduction, and business-driven data storytelling.

---

##  License

This project is open-source and available for educational and portfolio use.
