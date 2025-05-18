import numpy as np
from typing import List, Optional, Dict
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

class MLModel:
    def __init__(self):
        self.model: Optional[MLPClassifier] = None
        self.train_accuracies: List[float] = []
        self.test_accuracies: List[float] = []
        self.X_test: Optional[np.ndarray] = None
        self.y_test: Optional[np.ndarray] = None

    def train(self, X_train: np.ndarray, X_test: np.ndarray, 
              y_train: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        self.X_test, self.y_test = X_test, y_test

        try:
            sm = SMOTE(random_state=42)
            X_train, y_train = sm.fit_resample(X_train, y_train)
        except Exception as e:
            raise ValueError(f"Ошибка балансировки: {e}")

        self.model = MLPClassifier(
            hidden_layer_sizes=(50, 20),
            max_iter=1,
            warm_start=True,
            random_state=42
        )

        best_test_acc = 0
        no_improve = 0

        for epoch in range(1, 101):
            self.model.fit(X_train, y_train)

            train_acc = accuracy_score(y_train, self.model.predict(X_train))
            test_acc = accuracy_score(y_test, self.model.predict(X_test))

            self.train_accuracies.append(train_acc)
            self.test_accuracies.append(test_acc)

            if test_acc > best_test_acc + 0.001:
                best_test_acc = test_acc
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= 10:
                break

        cv_model = MLPClassifier(hidden_layer_sizes=(50, 20), max_iter=200, random_state=42)
        cv_scores = cross_val_score(cv_model, X_train, y_train, cv=5, scoring='accuracy')

        return {
            'cv_mean': np.mean(cv_scores),
            'cv_std': np.std(cv_scores),
            'early_stopped': no_improve >= 10
        }

    def predict(self, input_data: List, scaler: StandardScaler) -> int:
        scaled = scaler.transform([input_data])
        return self.model.predict(scaled)[0]
