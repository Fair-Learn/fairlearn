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
from themis_ml.datasets import german_credit, census_income
import asyncio

class FairnessAnalyzerr:
    def __init__(self, test_size=0.3, random_state=82, privileged_group="telephone"):
        self.test_size = test_size
        self.random_state = random_state
        self.privileged_group = privileged_group
        self.german_credit_df = german_credit(raw=True)
        self.census_income_df = census_income(raw=True)
        self.labels = pd.Series(self.german_credit_df['credit_risk'].values)
        self.features = pd.get_dummies(self.german_credit_df.drop('credit_risk', axis=1))
        for col in self.features.columns:
            if self.features[col].dtype == 'bool':
                self.features[col] = self.features[col].astype(int)
        self.protected_attribute = pd.Series(self.german_credit_df[self.privileged_group].values, dtype=int)
        self.scaler = StandardScaler()
        self.X_train, self.X_test, self.y_train, self.y_test, self.s_train, self.s_test = train_test_split(self.features, self.labels, self.protected_attribute, test_size=self.test_size, random_state=self.random_state)
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        self.results = {}

    





    async def train_and_evaluate(self, X_train_scaled, X_test_scaled, y_train, y_test, s_test):
        classifiers = {
            "SVC": SVC(probability=True),
            "Random Forest": RandomForestClassifier(random_state=self.random_state),
            "Logistic Regression": LogisticRegression(max_iter=2000, random_state=self.random_state),
            "XGBClassifier": XGBClassifier(random_state=self.random_state)
        }

        for clf_name, clf in classifiers.items():
            print(f"Processing: {clf_name}")
            clf.fit(X_train_scaled, y_train)
            predictions = clf.predict(X_test_scaled)
            prob_predictions = clf.predict_proba(X_test_scaled)[:, 1] if hasattr(clf, "predict_proba") else clf.decision_function(X_test_scaled)

            accuracy = accuracy_score(y_test, predictions)
            balanced_acc = balanced_accuracy_score(y_test, predictions)
            auc_roc = roc_auc_score(y_test, prob_predictions)
            report = classification_report(y_test, predictions, output_dict=True)
            conf_matrix = confusion_matrix(y_test, predictions)

            dataset_true = BinaryLabelDataset(df=pd.concat([self.X_test.reset_index(drop=True),
                                                            pd.DataFrame(y_test.values, columns=['credit_risk']),
                                                            pd.DataFrame(s_test.values, columns=[self.privileged_group])], axis=1),
                                              label_names=['credit_risk'],
                                              protected_attribute_names=[self.privileged_group])

            dataset_pred = dataset_true.copy()
            dataset_pred.labels = predictions.reshape(-1, 1)
            metric = ClassificationMetric(dataset_true, dataset_pred,
                                          unprivileged_groups=[{self.privileged_group: 0}],
                                          privileged_groups=[{self.privileged_group: 1}])

            self.results[clf_name] = {
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

    async def run_analysis(self):
        await self.train_and_evaluate(self.X_train_scaled, self.X_test_scaled, self.y_train, self.y_test, self.s_test)

    def display_results(self):
        for name, result in self.results.items():
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

# Usage example
if __name__ == "__main__":
    analyzer = FairnessAnalyzerr()
    asyncio.run(analyzer.run_analysis())
    analyzer.display_results()
