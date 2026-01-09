# Imarticus Data Science Internship – Assessment (Cohort 6)

## Goal

Build an **AI-Powered Data Storyteller** that:
- Accepts any CSV upload  
- Performs automated EDA (summary stats, correlations, value counts)  
- Shows plain-English insights & meaningful plots  
- Predicts employee attrition (when dataset matches the HR schema)  
- Generates an executive summary report (PDF / Word)  
- Runs as an interactive **Streamlit** dashboard

## Project Structure




## How to Run

### Setup environment

# From project root
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
pip install -r requirements.txt


# Launch Streamlit dashboard
cd dashboard
streamlit run app.py


# Use the App 

1. Upload a CSV file

2. Check EDA tab for summary, correlations, & plots

3. (If HR Attrition dataset) → go to Predictions tab

4. See Model Insights + download report as PDF/Word



# Libraries Used

1. Data: pandas, numpy

2. Visualization: matplotlib, seaborn, plotly

3. ML: scikit-learn, joblib

4. Dashboard: streamlit

5. Reports: fpdf, python-docx