
![sales_forecast](https://github.com/user-attachments/assets/8fb4b24b-f790-4fbb-bb46-3dd39ee36481)


# 🛒 Rossmann Sales Forecasting  

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)  
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)  
![Scikit-Learn](https://img.shields.io/badge/ML-ScikitLearn-orange)  
![XGBoost](https://img.shields.io/badge/Model-XGBoost-green)  
![LightGBM](https://img.shields.io/badge/Model-LightGBM-lightgreen)  
![Streamlit](https://img.shields.io/badge/App-Streamlit-red)  
![Status](https://img.shields.io/badge/Project-Complete-brightgreen)  

This project builds a **machine learning pipeline to forecast daily sales** for over 1,000 Rossmann stores across Europe.  
The goal is to **predict sales six weeks into the future**, helping management plan for **inventory, staffing, and promotions**.  

---

## 📂 Project Structure  

```
sales_forecast_project/
│── data/
│   ├── train.csv               # Training dataset
│   ├── test.csv                # Test dataset
│   ├── store.csv               # Store metadata
│   └── sample_submission.csv   # Submission template
│
│── models/
│   ├── final_model.joblib      # Final trained model
│   └── gbm_model.txt           # LightGBM model
│
│── Rossmann_forecasting_enhanced.ipynb   # Jupyter notebook
│── streamlit_app.py                      # Streamlit web app for predictions
│── README.md
│── requirements.txt
```

---

## 📊 Dataset  

- **train.csv** → Historical daily sales for Rossmann stores (2013–2015).  
- **test.csv** → Data for the forecast horizon.  
- **store.csv** → Metadata about stores (e.g., store type, assortment, competition info).  
- **sample_submission.csv** → Required format for predictions.  

Key columns include:  
- `Store`: Unique ID for each store  
- `Sales`: Target variable → daily sales  
- `Customers`: Number of customers  
- `Open`: Store open flag (0 = closed)  
- `Promo`: Whether the store had a promotion that day  
- `SchoolHoliday`: Whether the day was a holiday  

---

## 🔎 Workflow  

1. **Data Exploration & Cleaning**  
   - Checked missing values and store-wise distributions  
   - Explored correlations between promotions, holidays, and sales  

2. **Feature Engineering**  
   - Date decomposition (year, month, week, day, etc.)  
   - Promo & holiday encoding  
   - Competition distance and promo duration features  
   - Log transformation of skewed variables  

3. **Model Training**  
   - Baseline models: Linear Regression, Random Forest  
   - Advanced models: **XGBoost** and **LightGBM**  
   - Hyperparameter tuning via cross-validation  

4. **Model Evaluation**  
   - Metric: **Root Mean Squared Percentage Error (RMSPE)**  
   - XGBoost and LightGBM gave best performance  
   - Feature importance showed `Promo`, `CompetitionDistance`, and `Month` as key drivers  

5. **Forecasting & Visualization**  
   - Predicted store-level daily sales  
   - Compared predicted vs. actual on validation data  

6. **Deployment**  
   - `streamlit_app.py` provides a web interface to forecast sales for any store  
   - Users can input parameters and view predicted sales  
  
![sales_forecast](https://github.com/user-attachments/assets/bf5b7868-c4dc-423f-83bd-b1344fc9d5ea)


---

## 📈 Key Insights  

- Promotions significantly drive sales uplift  
- Sales vary seasonally, peaking during December  
- Stores with closer competitors have lower sales  
- Some stores are highly promotion-dependent  

---

## 🚀 How to Run  

### 1. Clone repo  
```bash
git clone https://github.com/yourusername/sales-forecast-rossmann.git
cd sales-forecast-rossmann
```

### 2. Install dependencies  
```bash
pip install -r requirements.txt
```

### 3. Run Notebook  
```bash
jupyter notebook Rossmann_forecasting_enhanced.ipynb
```

### 4. Run Streamlit App  
```bash
streamlit run streamlit_app.py
```

---

## 📌 Future Improvements  

- Incorporate **external data** (weather, holidays, macroeconomic trends)  
- Try **deep learning models** (LSTM, Temporal Fusion Transformer)  
- Deploy a full **cloud API** for sales forecasting  

---

## 🏆 Author  

**Paul Egeonu**  
_Data Analyst | Data Scientist_  
[LinkedIn](https://www.linkedin.com/in/paul-egeonu) | [GitHub](https://github.com/Paul-Egeonu)  
