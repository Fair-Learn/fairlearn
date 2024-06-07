from themis_ml.datasets import german_credit, census_income
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
import asyncio


class FairnessAnalyzer:



    def __init__(self, test_size=0.3, random_state=82, privileged_group="telephone"):
        self.test_size = test_size
        self.random_state = random_state
        self.selected_privileged_groups = privileged_group
        self.german_credit_df = german_credit(raw=True)
        self.census_income_df = census_income(raw=True)
        self.labels = pd.Series(self.german_credit_df['credit_risk'].values)
        self.features = pd.get_dummies(self.german_credit_df.drop('credit_risk', axis=1))

        for col in self.features.columns:
            if self.features[col].dtype == 'bool':
                self.features[col] = self.features[col].astype(int)

        self.protected_attribute = pd.Series(self.german_credit_df[self.selected_privileged_groups].values, dtype=int)
        self.results = {}
        self.scaler = StandardScaler()
        self.X_train, self.X_test, self.y_train, self.y_test, self.s_train, self.s_test = train_test_split(self.features, self.labels, self.protected_attribute, test_size=self.test_size, random_state=self.random_state)
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        self.results = {}
        self.classifiers = {
            "SVC": SVC(probability=True),
            "Random Forest": RandomForestClassifier(random_state=random_state),
            "Logistic Regression": LogisticRegression(max_iter=2000, random_state=random_state),
            "XGBClassifier": XGBClassifier(random_state=random_state)
        }




    async def loop(self):

        for clf_name, clf in self.classifiers.items():
            print(f"Processing: {clf_name}")

            clf.fit(self.X_train_scaled, self.y_train)
            predictions = clf.predict(self.X_test_scaled)
            prob_predictions = clf.predict_proba(self.X_test_scaled)[:, 1] if hasattr(clf, "predict_proba") else clf.decision_function(self.X_test_scaled)

            accuracy = accuracy_score(self.y_test, predictions)
            balanced_acc = balanced_accuracy_score(self.y_test, predictions)
            auc_roc = roc_auc_score(self.y_test, prob_predictions)
            report = classification_report(self.y_test, predictions, output_dict=True)
            conf_matrix = confusion_matrix(self.y_test, predictions)

            dataset_true = BinaryLabelDataset(df=pd.concat([self.X_test.reset_index(drop=True),
                                                            pd.DataFrame(self.y_test.values, columns=['credit_risk']),
                                                            pd.DataFrame(self.s_test.values, columns=[self.selected_privileged_groups])], axis=1),
                                            label_names=['credit_risk'],
                                            protected_attribute_names=[self.selected_privileged_groups])

            dataset_pred = dataset_true.copy()
            dataset_pred.labels = predictions.reshape(-1, 1)
            metric = ClassificationMetric(dataset_true, dataset_pred,
                                        unprivileged_groups=[{self.selected_privileged_groups: 0}],
                                        privileged_groups=[{self.selected_privileged_groups: 1}])

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

        # return self.results

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




async def main():
    analyzer = FairnessAnalyzer(0.2, 82, "telephone")

    await analyzer.loop()

#     # analyzer = SecondPart()

#     # await analyzer.loops()

if __name__ == "__main__":
    asyncio.run(main())