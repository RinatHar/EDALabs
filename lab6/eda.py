# eda.py
import pandas as pd
from ydata_profiling import ProfileReport
import os

def load_data():
    data_path = 'data/online_shoppers.csv'
    
    if os.path.exists(data_path):
        return pd.read_csv(data_path)
    
    from ucimlrepo import fetch_ucirepo
    dataset = fetch_ucirepo(id=468)
    
    X = dataset.data.features
    y = dataset.data.targets
    df = X.copy()
    df['Revenue'] = y
    
    os.makedirs('data', exist_ok=True)
    df.to_csv(data_path, index=False)
    
    return df

def create_eda_report():
    df = load_data()
    
    profile = ProfileReport(
        df,
        title='Online Shoppers Report',
        explorative=True,
        minimal=False
    )
    
    report_path = 'data/online_shoppers_report.html'
    profile.to_file(report_path)
    
    return report_path

def get_dataset_info():
    df = load_data()
    return {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.astype(str).to_dict(),
        'missing': df.isnull().sum().to_dict(),
        'target_dist': df['Revenue'].value_counts().to_dict()
    }

if __name__ == '__main__':
    create_eda_report()