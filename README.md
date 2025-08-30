# Walmart Sales Forecasting with Holiday Markdown Effects Analysis

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Machine Learning](https://img.shields.io/badge/Machine-Learning-orange)
![Time Series](https://img.shields.io/badge/Time-Series-green)
![Retail Analytics](https://img.shields.io/badge/Retail-Analytics-purple)

## 📊 Project Overview

This machine learning project predicts weekly sales for 45 Walmart stores across multiple departments, with special focus on holiday periods and markdown (promotional) effects. The solution addresses key retail challenges including limited holiday data, complex promotional impacts, and department-level forecasting.

## 🎯 Business Problem

Retailers face significant challenges in predicting sales during holiday periods due to:
- Limited historical data for specific holiday events
- Complex interactions between markdowns and sales patterns
- Varying effects across different product departments
- Need for accurate inventory planning and staffing

## 📁 Project Structure

```
walmart-sales-forecasting/
├── notebooks/
│   └── walmart_analysis.ipynb    # Comprehensive EDA and modeling
├── data/                         # Dataset files (ignored in git)
├── src/                          # Source code modules
├── models/                       # Trained model files
├── reports/                      # Analysis reports and visualizations
├── README.md                     # Project documentation
└── requirements.txt              # Python dependencies
```

## 🛠️ Technical Stack

- **Python 3.9+**
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-learn, XGBoost, LightGBM
- **Visualization**: Matplotlib, Seaborn
- **Time Series Analysis**: Statsmodels
- **Notebooks**: Jupyter

## 📈 Key Features

### 🔍 Data Analysis
- Comprehensive exploratory data analysis (EDA)
- Holiday sales pattern identification
- Markdown effectiveness quantification
- Store and department performance comparisons

### ⚙️ Feature Engineering
- Time-based features (year, month, week, holidays)
- Lag variables and rolling statistics
- Markdown impact metrics
- Department-store interactions

### 🤖 Machine Learning
- Multiple model approaches (XGBoost, LightGBM, Random Forest)
- Time-series cross-validation
- Weighted Mean Absolute Error (WMAE) optimization
- Hyperparameter tuning

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- Git
- Conda (recommended) or pip

### Installation

```bash
# Clone the repository
git clone https://github.com/Yisakwondwosen/walmart-sales-forecasting.git
cd walmart-sales-forecasting

# Create conda environment
conda create -n walmart-forecasting python=3.9
conda activate walmart-forecasting

# Install dependencies
pip install -r requirements.txt
```

### Usage

```bash
# Launch Jupyter notebook for exploratory analysis
jupyter notebook notebooks/walmart_analysis.ipynb

# Or run specific analysis scripts
python src/data_processing.py
```

## 📊 Dataset

The project uses the [Walmart Recruiting - Store Sales Forecasting](https://www.kaggle.com/competitions/walmart-recruiting-store-sales-forecasting) dataset from Kaggle, which includes:

- **train.csv**: Historical sales data (2010-2012)
- **test.csv**: Validation data
- **features.csv**: Store features, markdowns, economic indicators
- **stores.csv**: Store metadata

### Data Citation
```bibtex
@misc{walmart-recruiting-store-sales-forecasting,
    author = {Walmart Competition Admin and Will Cukierski},
    title = {Walmart Recruiting - Store Sales Forecasting},
    year = {2014},
    howpublished = {\url{https://kaggle.com/competitions/walmart-recruiting-store-sales-forecasting}},
    note = {Kaggle}
}
```

## 📈 Key Findings

### 🎯 Holiday Impact
- Holiday weeks show 2.3x higher sales on average compared to non-holiday weeks
- Thanksgiving and Christmas periods account for 30% of annual sales
- Markdown effectiveness varies significantly by department

### 🏪 Store Performance
- Store Type A shows highest average sales per square foot
- Geographical location significantly impacts sales patterns
- Size and type combinations reveal optimal store configurations

### 📊 Markdown Effectiveness
- MarkDown1 shows strongest correlation with sales increase
- Electronics departments most responsive to promotions
- Optimal markdown timing varies by product category

## 🎯 Model Performance

- **Evaluation Metric**: Weighted Mean Absolute Error (WMAE)
- **Holiday Weight**: 5x penalty for holiday week errors
- **Baseline Improvement**: X% improvement over historical averages
- **Key Strength**: Accurate holiday period predictions

## 🤝 Contributing

This project is open for exploration and learning. Feel free to:

1. Fork the repository
2. Experiment with different modeling approaches
3. Extend the analysis to other retail domains
4. Suggest improvements via issues or pull requests

## 📝 License

This project is created for educational and portfolio purposes. The dataset belongs to Walmart and Kaggle. Please respect the original data source terms and conditions.

## 👨‍💻 Author

**Yisakwondwosen**  
- GitHub: [@Yisakwondwosen](https://github.com/Yisakwondwosen)
- Upwork: [Data Science & Machine Learning Profile](https://www.upwork.com/freelancers/~)

## 🔗 Useful Links

- [Kaggle Competition Page](https://www.kaggle.com/competitions/walmart-recruiting-store-sales-forecasting)
- [Project Blog Post](https://yourblog.com/walmart-sales-forecasting) *(optional)*
- [LinkedIn Article](https://linkedin.com/in/yourprofile) *(optional)*

## 🎯 Future Enhancements

- [ ] Real-time prediction API
- [ ] Dashboard visualization
- [ ] Additional economic indicators
- [ ] Cross-chain retail comparisons
- [ ] Deep learning approaches (LSTM, Transformers)

---

⭐ **If you find this project useful, please give it a star on GitHub!**
