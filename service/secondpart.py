import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, balanced_accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric
from themis_ml.preprocessing.relabelling import Relabeller
from themis_ml.datasets import german_credit, census_income
import asyncio

# Parameters
TEST_SIZE = 0.3
RANDOM_STATE = 82
PRIVILEGED_GROUP1 = "foreign_worker"
PRIVILEGED_GROUP2 = "telephone"
SELECTED_PRIVILEGED_GROUP = PRIVILEGED_GROUP2

class DataProcessor:
    def __init__(self, data, label_col, privileged_group):
        self.data = data
        self.label_col = label_col
        self.privileged_group = privileged_group
        self.labels = pd.Series(self.data[self.label_col].values)
        self.features = pd.get_dummies(self.data.drop(self.label_col, axis=1))
        self.protected_attribute = pd.Series(self.data[self.privileged_group].values, dtype=int)

        # Ensure all boolean columns are converted to integers
        for col in self.features.columns:
            if self.features[col].dtype == 'bool':
                self.features[col] = self.features[col].astype(int)

    def split_data(self):
        return train_test_split(
            self.features, self.labels, self.protected_attribute,
            test_size=TEST_SIZE, random_state=RANDOM_STATE
        )

class ModelEvaluator:
    def __init__(self, X_train, X_test, y_train, y_test, s_train, s_test, scaler):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.s_train = s_train
        self.s_test = s_test
        self.scaler = scaler

    async def run_relabeller(self):
        relabeller = Relabeller()
        relabeller.fit(self.X_train, self.y_train, self.s_train)
        y_train_transformed = relabeller.transform(self.X_train)
        return self.X_train, y_train_transformed

    async def train_and_evaluate(self, method_name, method_func):
        X_train_transformed, y_train_transformed = await method_func()

        X_train_transformed_scaled = self.scaler.fit_transform(X_train_transformed)
        X_test_scaled = self.scaler.transform(self.X_test)

        classifiers = {
            "SVC": SVC(probability=True),
            "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE),
            "Logistic Regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
            "XGBClassifier": XGBClassifier(random_state=RANDOM_STATE)
        }

        results = {}
        for clf_name, clf in classifiers.items():
            clf.fit(X_train_transformed_scaled, y_train_transformed)
            predictions = clf.predict(X_test_scaled)
            prob_predictions = clf.predict_proba(X_test_scaled)[:, 1] if hasattr(clf, "predict_proba") else clf.decision_function(X_test_scaled)

            accuracy = accuracy_score(self.y_test, predictions)
            balanced_acc = balanced_accuracy_score(self.y_test, predictions)
            auc_roc = roc_auc_score(self.y_test, prob_predictions)
            report = classification_report(self.y_test, predictions, output_dict=True)
            conf_matrix = confusion_matrix(self.y_test, predictions)

            dataset_true = BinaryLabelDataset(df=pd.concat([self.X_test.reset_index(drop=True),
                                                            pd.DataFrame(self.y_test.values, columns=['credit_risk']),
                                                            pd.DataFrame(self.s_test.values, columns=[SELECTED_PRIVILEGED_GROUP])], axis=1),
                                              label_names=['credit_risk'],
                                              protected_attribute_names=[SELECTED_PRIVILEGED_GROUP])

            dataset_pred = dataset_true.copy()
            dataset_pred.labels = predictions.reshape(-1, 1)
            metric = ClassificationMetric(dataset_true, dataset_pred,
                                          unprivileged_groups=[{SELECTED_PRIVILEGED_GROUP: 0}],
                                          privileged_groups=[{SELECTED_PRIVILEGED_GROUP: 1}])

            results[f"{method_name} - {clf_name}"] = {
                "accuracy": accuracy,
                "balanced_accuracy": balanced_acc,
                "auc_roc": auc_roc,
                "report": report,
                "disparate_impact": metric.disparate_impact(),
                "statistical_parity_difference": metric.statistical_parity_difference(),
                "equal_opportunity_difference": metric.equal_opportunity_difference(),
                "average_odds_difference": metric.average_odds_difference(),
                "theil_index": metric.theil_index(),
                "conf_matrix": conf_matrix
            }

        return results

async def main():
    german_credit_df = german_credit(raw=True)
    data_processor = DataProcessor(german_credit_df, 'credit_risk', SELECTED_PRIVILEGED_GROUP)
    X_train, X_test, y_train, y_test, s_train, s_test = data_processor.split_data()
    scaler = StandardScaler()

    model_evaluator = ModelEvaluator(X_train, X_test, y_train, y_test, s_train, s_test, scaler)

    methods = {
        "Relabeller": model_evaluator.run_relabeller
    }

    overall_results = {}
    for method_name, method_func in methods.items():
        method_results = await model_evaluator.train_and_evaluate(method_name, method_func)
        overall_results.update(method_results)

    for name, result in overall_results.items():
        print(f"\n{name}")
        print(f"Accuracy: {result['accuracy']}")
        print(f"Balanced Accuracy: {result['balanced_accuracy']}")
        print(f"AUC-ROC: {result['auc_roc']}")
        print(f"Classification Report:\n {pd.DataFrame(result['report']).transpose()}")
        print(f"Disparate Impact: {result['disparate_impact']}")
        print(f"Statistical Parity Difference: {result['statistical_parity_difference']}")
        print(f"Equal Opportunity Difference: {result['equal_opportunity_difference']}")
        print(f"Average Odds Difference: {result['average_odds_difference']}")
        print(f"Theil Index: {result['theil_index']}")

        plt.figure(figsize=(8, 6))
        sns.heatmap(result['conf_matrix'], annot=True, fmt="d", cmap="Blues", xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
        plt.title(f'Confusion Matrix - {name}')
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        plt.show()

if __name__ == "__main__":
    asyncio.run(main())
