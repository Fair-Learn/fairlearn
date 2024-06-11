import pandas as pd
import numpy as np
import asyncio
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, balanced_accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
# from aequitas.flow.methods.preprocessing.label_flipping import LabelFlipping
# from aequitas.flow.methods.preprocessing.data_repairer import DataRepairer
# from aequitas.flow.methods.preprocessing.prevalence_sample import PrevalenceSampling
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric
from themis_ml.preprocessing.relabelling import Relabeller
from themis_ml.datasets import german_credit, census_income


class FairnessComparison:
    def __init__(self, test_size=0.3, random_state=82, privileged_group="telephone"):
        self.test_size = test_size
        self.random_state = random_state
        self.privileged_group = privileged_group

        self.german_credit_df = german_credit(raw=True)
        self.census_income_df = census_income(raw=True)

        # Prepare labels, features, and protected attribute
        self.labels = pd.Series(self.german_credit_df['credit_risk'].values)
        self.features = pd.get_dummies(self.german_credit_df.drop('credit_risk', axis=1))

        # Ensure all boolean columns are converted to integers
        for col in self.features.columns:
            if self.features[col].dtype == 'bool':
                self.features[col] = self.features[col].astype(int)

        self.protected_attribute = pd.Series(self.german_credit_df[self.privileged_group].values, dtype=int)

        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test, self.s_train, self.s_test = train_test_split(
            self.features, self.labels, self.protected_attribute, test_size=self.test_size, random_state=self.random_state)

        # Standardize the data
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)

        self.methods = {
            # "Label Flipping": self.run_label_flipping,
            # "Data Repairer": self.run_data_repairer,
            # "Prevalence Sampling": self.run_prevalence_sampling,
            "Relabeller": self.run_relabeller
        }

        self.classifiers = {
            "SVC": SVC(probability=True),
            "Random Forest": RandomForestClassifier(random_state=self.random_state),
            "Logistic Regression": LogisticRegression(max_iter=2000, random_state=self.random_state),
            "XGBClassifier": XGBClassifier(random_state=self.random_state)
        }

    async def run_label_flipping(self, X_train, y_train, s_train):
        flipper = LabelFlipping(max_flip_rate=0.2, fair_ordering=True)
        flipper.fit(X_train, y_train, s_train)
        X_train_transformed, y_train_transformed, _ = flipper.transform(X_train, y_train, s_train)
        return X_train_transformed, y_train_transformed

    async def run_data_repairer(self, X_train, y_train, s_train):
        repairer = DataRepairer(repair_level=1.0)
        repairer.fit(X_train, y_train, s_train)
        X_train_transformed, y_train_transformed, _ = repairer.transform(X_train, y_train, s_train)
        return X_train_transformed, y_train_transformed

    async def run_prevalence_sampling(self, X_train, y_train, s_train):
        sampler = PrevalenceSampling(alpha=1, strategy="undersample", s_ref="global")
        sampler.fit(X_train, y_train, s_train)
        X_train_transformed, y_train_transformed, _ = sampler.transform(X_train, y_train, s_train)
        return X_train_transformed, y_train_transformed

    async def run_relabeller(self, X_train, y_train, s_train):
        relabeller = Relabeller()
        relabeller.fit(X_train, y_train, s_train)
        y_train_transformed = relabeller.transform(X_train)
        return X_train, y_train_transformed

    async def process_method(self, name, method):
        print(f"Processing: {name}")

        # Reload the original X and y data for each method
        X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
            self.features, self.labels, self.protected_attribute, test_size=self.test_size, random_state=self.random_state)

        # Apply the method
        X_train_transformed, y_train_transformed = await method(X_train, y_train, s_train)

        X_train_transformed_scaled = self.scaler.fit_transform(X_train_transformed)
        X_test_scaled = self.scaler.transform(X_test)  # Ensure the test data is scaled with the same scaler

        results = {}

        for clf_name, clf in self.classifiers.items():
            clf.fit(X_train_transformed_scaled, y_train_transformed)
            predictions = clf.predict(X_test_scaled)
            prob_predictions = clf.predict_proba(X_test_scaled)[:, 1] if hasattr(clf, "predict_proba") else clf.decision_function(X_test_scaled)

            accuracy = accuracy_score(y_test, predictions)
            balanced_acc = balanced_accuracy_score(y_test, predictions)
            auc_roc = roc_auc_score(y_test, prob_predictions)
            report = classification_report(y_test, predictions, output_dict=True)
            conf_matrix = confusion_matrix(y_test, predictions)

            dataset_true = BinaryLabelDataset(df=pd.concat([X_test.reset_index(drop=True),
                                                            pd.DataFrame(y_test.values, columns=['credit_risk']),
                                                            pd.DataFrame(s_test.values, columns=[self.privileged_group])], axis=1),
                                              label_names=['credit_risk'],
                                              protected_attribute_names=[self.privileged_group])

            dataset_pred = dataset_true.copy()
            dataset_pred.labels = predictions.reshape(-1, 1)
            metric = ClassificationMetric(dataset_true, dataset_pred,
                                          unprivileged_groups=[{self.privileged_group: 0}],
                                          privileged_groups=[{self.privileged_group: 1}])

            results[f"{name} - {clf_name}"] = {
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

    async def run_all_methods(self):
        results = {}
        tasks = []
        for name, method in self.methods.items():
            tasks.append(asyncio.create_task(self.process_method(name, method)))

        all_results = await asyncio.gather(*tasks)
        
        for task_results in all_results:
            results.update(task_results)

        # Comparison of all methods
        for name, result in results.items():
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

# Example usage
if __name__ == "__main__":
    comparison = FairnessComparison(test_size=0.3, random_state=82, privileged_group="telephone")
    asyncio.run(comparison.run_all_methods())
