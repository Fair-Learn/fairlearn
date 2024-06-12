import asyncio
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
from aif360.algorithms.preprocessing import DisparateImpactRemover, Reweighing
from aif360.metrics import ClassificationMetric
from themis_ml.datasets import german_credit, census_income

class ThirdPart:
    def __init__(self, test_size=0.3, random_state=82, privileged_group="telephone"):
        self.test_size = test_size
        self.random_state = random_state
        self.privileged_group = privileged_group
        self.results = {}

    async def load_data(self):
        self.german_credit_df = german_credit(raw=True)
        self.census_income_df = census_income(raw=True)

    def prepare_data(self):
        self.labels = self.german_credit_df['credit_risk']
        self.features = self.german_credit_df.drop('credit_risk', axis=1)

        for col in self.features.columns:
            if self.features[col].dtype == 'bool':
                self.features[col] = self.features[col].astype(int)

        self.protected_attribute = self.german_credit_df[self.privileged_group].astype(int)
        self.X_train, self.X_test, self.y_train, self.y_test, self.s_train, self.s_test = train_test_split(
            self.features, self.labels, self.protected_attribute, test_size=self.test_size, random_state=self.random_state)

        categorical_columns = ['status_of_existing_checking_account', 'credit_history', 'purpose', 'savings_account/bonds',
                               'present_employment_since', 'personal_status_and_sex', 'other_debtors/guarantors', 'property',
                               'other_installment_plans', 'housing', 'job']
        self.X_encoded = pd.get_dummies(self.features, columns=categorical_columns)

    def run_disparate_impact_remover(self, X_encoded, labels):
        binary_label_dataset = BinaryLabelDataset(df=pd.concat([X_encoded, labels], axis=1),
                                                  label_names=['credit_risk'],
                                                  protected_attribute_names=[self.privileged_group])

        dir = DisparateImpactRemover(repair_level=1.0)
        repaired_bld = dir.fit_transform(binary_label_dataset)

        repaired_df, _ = repaired_bld.convert_to_dataframe()
        X = repaired_df.drop('credit_risk', axis=1)
        y = repaired_df['credit_risk']
        return X, y, None

    def run_reweighing(self, X_encoded, labels):
        binary_label_dataset = BinaryLabelDataset(df=pd.concat([X_encoded, labels], axis=1),
                                                  label_names=['credit_risk'],
                                                  protected_attribute_names=[self.privileged_group])

        rw = Reweighing(unprivileged_groups=[{self.privileged_group: 0}], privileged_groups=[{self.privileged_group: 1}])
        reweighed_dataset = rw.fit_transform(binary_label_dataset)

        X = reweighed_dataset.features
        y = reweighed_dataset.labels.ravel()
        weights = reweighed_dataset.instance_weights
        return X, y, weights

    async def train_and_evaluate(self):
        methods = {
            "Disparate Impact Remover": self.run_disparate_impact_remover,
            "Reweighing": self.run_reweighing
        }

        for method_name, method in methods.items():
            print(f"Processing: {method_name}")

            X_transformed, y_transformed, weights = method(self.X_encoded, self.labels)

            if weights is not None:
                X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(X_transformed, y_transformed, weights, test_size=self.test_size, random_state=self.random_state)
            else:
                X_train, X_test, y_train, y_test = train_test_split(X_transformed, y_transformed, test_size=self.test_size, random_state=self.random_state)
                w_train = None

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            models = {
                "Logistic Regression": LogisticRegression(max_iter=2000, random_state=self.random_state),
                "SVC": SVC(probability=True, random_state=self.random_state),
                "Random Forest": RandomForestClassifier(random_state=self.random_state),
                "XGBClassifier": XGBClassifier(random_state=self.random_state)
            }

            for model_name, model in models.items():
                print(f"Processing: {method_name} - {model_name}")

                if method_name == "Reweighing":
                    model.fit(X_train_scaled, y_train, sample_weight=w_train)
                else:
                    model.fit(X_train_scaled, y_train)

                predictions = model.predict(X_test_scaled)
                accuracy = accuracy_score(y_test, predictions)
                auc_roc = roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:, 1])
                balanced_acc = balanced_accuracy_score(y_test, predictions)
                conf_matrix = confusion_matrix(y_test, predictions)
                report = classification_report(y_test, predictions, output_dict=True)

                dataset_true = BinaryLabelDataset(df=pd.DataFrame(np.column_stack([X_test, y_test, self.s_test]),
                                                                  columns=list(self.X_encoded.columns) + ['credit_risk', self.privileged_group]),
                                                  label_names=['credit_risk'],
                                                  protected_attribute_names=[self.privileged_group])

                dataset_pred = dataset_true.copy()
                dataset_pred.labels = predictions.reshape(-1, 1)

                metric = ClassificationMetric(dataset_true, dataset_pred,
                                              unprivileged_groups=[{self.privileged_group: 0}],
                                              privileged_groups=[{self.privileged_group: 1}])

                self.results[f"{method_name} - {model_name}"] = {
                    "accuracy": accuracy,
                    "auc_roc": auc_roc,
                    "balanced_accuracy": balanced_acc,
                    "report": report,
                    "disparate_impact": metric.disparate_impact(),
                    "statistical_parity_difference": metric.statistical_parity_difference(),
                    "equal_opportunity_difference": metric.equal_opportunity_difference(),
                    "average_odds_difference": metric.average_odds_difference(),
                    "theil_index": metric.theil_index(),
                    "conf_matrix": conf_matrix
                }

    async def run_experiment(self):
        await self.load_data()
        self.prepare_data()
        await self.train_and_evaluate()
        return self.results

    def resultpart(self):
        return self.results

    def display_results(self):
        for name, result in self.results.items():
            print(f"\n{name}")
            print(f"Accuracy: {result['accuracy']}")
            print(f"AUC-ROC: {result['auc_roc']}")
            print(f"Balanced Accuracy: {result['balanced_accuracy']}")
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

# Running the experiment
# experiment = ThirdPart()
# asyncio.run(experiment.run_experiment())
# experiment.display_results()
