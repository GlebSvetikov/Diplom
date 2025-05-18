import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from typing import Dict, List, Optional, Tuple

class DataModel:
    def __init__(self):
        self.training_data: Optional[pd.DataFrame] = None
        self.cadets_data: pd.DataFrame = pd.DataFrame()
        self.scaler: Optional[StandardScaler] = None
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.feature_order: List[str] = []

    def load_training_data(self, file_path: str) -> None:
        self.training_data = pd.read_excel(file_path)

    def preprocess_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        df = self.training_data.copy()
        df['target'] = df['Уровень успешности в ЛП2'].apply(
            lambda x: 1 if x.strip() == 'не отчислен' else 0
        )
        df.drop(columns=['Уровень успешности в ЛП2'], inplace=True)

        self.feature_order = df.drop(columns=['target']).columns.tolist()

        for col in self.feature_order:
            if df[col].dtype == 'object':
                df[col] = df[col].fillna('NA')
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                self.label_encoders[col] = le
            else:
                df[col] = df[col].fillna(df[col].mean())

        X = df.drop(columns=['target'])
        y = df['target']

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        return train_test_split(X_scaled, y, test_size=0.15, random_state=42)

    def add_cadet(self, cadet_data: Dict[str, str]) -> None:
        cadet_data["Уровень успешности в ЛП2"] = "не оценен"
        new_data = pd.DataFrame([cadet_data])
        self.cadets_data = pd.concat([self.cadets_data, new_data], ignore_index=True)
        self._reorder_columns()

    def _reorder_columns(self) -> None:
        cols = self.cadets_data.columns.tolist()
        if 'Уровень успешности в ЛП2' in cols:
            new_order = ['Уровень успешности в ЛП2'] + [c for c in cols if c != 'Уровень успешности в ЛП2']
            self.cadets_data = self.cadets_data[new_order]
