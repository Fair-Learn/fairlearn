from themis_ml.datasets import german_credit, census_income

german_credit_df = german_credit(raw=True)
census_income_df = census_income(raw=True)

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

# Parameters
test_size = 0.3
random_state = 82
privileged_groups1 = "foreign_worker"
privileged_groups2 = "telephone"
selected_privileged_groups = privileged_groups2

# Prepare labels, features, and protected attribute
labels = pd.Series(german_credit_df['credit_risk'].values)
features = pd.get_dummies(german_credit_df.drop('credit_risk', axis=1))

# Ensure all boolean columns are converted to integers
for col in features.columns:
    if features[col].dtype == 'bool':
        features[col] = features[col].astype(int)

protected_attribute = pd.Series(german_credit_df[selected_privileged_groups].values, dtype=int)

# Split data
X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
    features, labels, protected_attribute, test_size=test_size, random_state=random_state)

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

results = {}

classifiers = {
    "SVC": SVC(probability=True),
    "Random Forest": RandomForestClassifier(random_state=random_state),
    "Logistic Regression": LogisticRegression(max_iter=2000, random_state=random_state),
    "XGBClassifier": XGBClassifier(random_state=random_state)
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

    dataset_true = BinaryLabelDataset(df=pd.concat([X_test.reset_index(drop=True),
                                                    pd.DataFrame(y_test.values, columns=['credit_risk']),
                                                    pd.DataFrame(s_test.values, columns=[selected_privileged_groups])], axis=1),
                                      label_names=['credit_risk'],
                                      protected_attribute_names=[selected_privileged_groups])

    dataset_pred = dataset_true.copy()
    dataset_pred.labels = predictions.reshape(-1, 1)
    metric = ClassificationMetric(dataset_true, dataset_pred,
                                  unprivileged_groups=[{selected_privileged_groups: 0}],
                                  privileged_groups=[{selected_privileged_groups: 1}])

    results[clf_name] = {
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

# Comparison of all models
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
from aequitas.flow.methods.preprocessing.label_flipping import LabelFlipping
from aequitas.flow.methods.preprocessing.data_repairer import DataRepairer
from aequitas.flow.methods.preprocessing.prevalence_sample import PrevalenceSampling
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric
from themis_ml.preprocessing.relabelling import Relabeller

# Parameters
test_size = 0.3
random_state = 82
privileged_groups1="foreign_worker"
privileged_groups2="telephone"
selected_privileged_groups = privileged_groups2

# Prepare labels, features, and protected attribute
labels = pd.Series(german_credit_df['credit_risk'].values)
features = pd.get_dummies(german_credit_df.drop('credit_risk', axis=1))

# Ensure all boolean columns are converted to integers
for col in features.columns:
    if features[col].dtype == 'bool':
        features[col] = features[col].astype(int)

protected_attribute = pd.Series(german_credit_df[selected_privileged_groups].values, dtype=int)

# Split data
X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
    features, labels, protected_attribute, test_size=test_size, random_state=random_state)

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

def run_label_flipping(X_train, y_train, s_train):
    flipper = LabelFlipping(max_flip_rate=0.2, fair_ordering=True)
    flipper.fit(X_train, y_train, s_train)
    X_train_transformed, y_train_transformed, _ = flipper.transform(X_train, y_train, s_train)
    return X_train_transformed, y_train_transformed

def run_data_repairer(X_train, y_train, s_train):
    repairer = DataRepairer(repair_level=1.0)
    repairer.fit(X_train, y_train, s_train)
    X_train_transformed, y_train_transformed, _ = repairer.transform(X_train, y_train, s_train)
    return X_train_transformed, y_train_transformed

def run_prevalence_sampling(X_train, y_train, s_train):
    sampler = PrevalenceSampling(alpha=1, strategy="undersample", s_ref="global")
    sampler.fit(X_train, y_train, s_train)
    X_train_transformed, y_train_transformed, _ = sampler.transform(X_train, y_train, s_train)
    return X_train_transformed, y_train_transformed

def run_relabeller(X_train, y_train, s_train):
    relabeller = Relabeller()
    relabeller.fit(X_train, y_train, s_train)
    y_train_transformed = relabeller.transform(X_train)
    return X_train, y_train_transformed

methods = {
    "Label Flipping": run_label_flipping,
    "Data Repairer": run_data_repairer,
    "Prevalence Sampling": run_prevalence_sampling,
    "Relabeller": run_relabeller
}

results = {}

# Train and evaluate models for each method and classifier
for name, method in methods.items():
    print(f"Processing: {name}")

    # Reload the original X and y data for each method
    X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
        features, labels, protected_attribute, test_size=test_size, random_state=random_state)

    # Apply the method
    X_train_transformed, y_train_transformed = method(X_train, y_train, s_train)

    X_train_transformed_scaled = scaler.fit_transform(X_train_transformed)
    X_test_scaled = scaler.transform(X_test)  # Ensure the test data is scaled with the same scaler

    classifiers = {
        "SVC": SVC(probability=True),
        "Random Forest": RandomForestClassifier(random_state=random_state),
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=random_state),
        "XGBClassifier": XGBClassifier(random_state=random_state)
    }

    for clf_name, clf in classifiers.items():
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
                                                        pd.DataFrame(s_test.values, columns=[selected_privileged_groups])], axis=1),
                                          label_names=['credit_risk'],
                                          protected_attribute_names=[selected_privileged_groups])

        dataset_pred = dataset_true.copy()
        dataset_pred.labels = predictions.reshape(-1, 1)
        metric = ClassificationMetric(dataset_true, dataset_pred,
                                      unprivileged_groups=[{selected_privileged_groups: 0}],
                                      privileged_groups=[{selected_privileged_groups: 1}])

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
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric

protected_attribute = selected_privileged_groups

# Prepare labels, features, and protected attribute
labels = german_credit_df['credit_risk']
features = german_credit_df.drop('credit_risk', axis=1)

categorical_columns = ['status_of_existing_checking_account', 'credit_history', 'purpose', 'savings_account/bonds',
                       'present_employment_since', 'personal_status_and_sex', 'other_debtors/guarantors', 'property',
                       'other_installment_plans', 'housing', 'job']
X_encoded = pd.get_dummies(features, columns=categorical_columns)

# Function to run Disparate Impact Remover
def run_disparate_impact_remover(X_encoded, labels):
    binary_label_dataset = BinaryLabelDataset(df=pd.concat([X_encoded, labels], axis=1),
                                              label_names=['credit_risk'],
                                              protected_attribute_names=[protected_attribute])

    dir = DisparateImpactRemover(repair_level=1.0)
    repaired_bld = dir.fit_transform(binary_label_dataset)

    repaired_df, _ = repaired_bld.convert_to_dataframe()
    X = repaired_df.drop('credit_risk', axis=1)
    y = repaired_df['credit_risk']
    return X, y, None  # Returning None for weights

# Function to run Reweighing
def run_reweighing(X_encoded, labels):
    binary_label_dataset = BinaryLabelDataset(df=pd.concat([X_encoded, labels], axis=1),
                                              label_names=['credit_risk'],
                                              protected_attribute_names=[protected_attribute])

    rw = Reweighing(unprivileged_groups=[{protected_attribute: 0}], privileged_groups=[{protected_attribute: 1}])
    reweighed_dataset = rw.fit_transform(binary_label_dataset)

    X = reweighed_dataset.features
    y = reweighed_dataset.labels.ravel()
    weights = reweighed_dataset.instance_weights
    return X, y, weights

methods = {
    "Disparate Impact Remover": run_disparate_impact_remover,
    "Reweighing": run_reweighing
}

results = {}

# Train and evaluate models for each method and classifier
for method_name, method in methods.items():
    print(f"Processing: {method_name}")

    if method_name == "Disparate Impact Remover":
        X_transformed, y_transformed, weights = method(X_encoded, labels)
    else:
        X_transformed, y_transformed, weights = method(X_encoded, labels)

    if weights is not None:
        X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(X_transformed, y_transformed, weights, test_size=test_size, random_state=random_state)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X_transformed, y_transformed, test_size=test_size, random_state=random_state)
        w_train = None  # Setting weights to None for non-Reweighing methods

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=random_state),
        "Support Vector Machine": SVC(probability=True, random_state=random_state),
        "Random Forest": RandomForestClassifier(random_state=random_state),
        "Gradient Boosting Machines": XGBClassifier(random_state=random_state)
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

        dataset_true = BinaryLabelDataset(df=pd.DataFrame(np.column_stack([X_test, y_test, s_test]),
                                                          columns=list(X_encoded.columns) + ['credit_risk', protected_attribute]),
                                          label_names=['credit_risk'],
                                          protected_attribute_names=[protected_attribute])

        dataset_pred = dataset_true.copy()
        dataset_pred.labels = predictions.reshape(-1, 1)

        metric = ClassificationMetric(dataset_true, dataset_pred,
                                      unprivileged_groups=[{protected_attribute: 0}],
                                      privileged_groups=[{protected_attribute: 1}])

        results[f"{method_name} - {model_name}"] = {
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

# Comparison of all methods
for name, result in results.items():
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

from ucimlrepo import fetch_ucirepo

# fetch dataset
statlog_german_credit_data = fetch_ucirepo(id=144) # German Credit Data
adult = fetch_ucirepo(id=2) # Adult Census

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
from sklearn.preprocessing import StandardScaler, LabelEncoder
from aif360.datasets import BinaryLabelDataset
from aif360.algorithms.preprocessing import DisparateImpactRemover, Reweighing
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from ucimlrepo import fetch_ucirepo

# Data (as pandas dataframes)
X = adult.data.features
y = adult.data.targets

# Metadata
print(adult.metadata)

# Variable information
print(adult.variables)

# Prepare labels, features, and protected attribute
labels = y['class'].apply(lambda x: 1 if x == '>50K' else 0)
features = X.drop('class', axis=1)

# Encode categorical columns
categorical_columns = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
features = pd.get_dummies(features, columns=categorical_columns)

# Set protected attribute
protected_attribute = 'sex_ Female'

# Function to run Disparate Impact Remover
def run_disparate_impact_remover(X_encoded, labels):
    binary_label_dataset = BinaryLabelDataset(df=pd.concat([X_encoded, labels], axis=1),
                                              label_names=['income'],
                                              protected_attribute_names=[protected_attribute])

    dir = DisparateImpactRemover(repair_level=1.0)
    repaired_bld = dir.fit_transform(binary_label_dataset)

    repaired_df, _ = repaired_bld.convert_to_dataframe()
    X = repaired_df.drop('income', axis=1)
    y = repaired_df['income']
    return X, y, None  # Returning None for weights

# Function to run Reweighing
def run_reweighing(X_encoded, labels):
    binary_label_dataset = BinaryLabelDataset(df=pd.concat([X_encoded, labels], axis=1),
                                              label_names=['income'],
                                              protected_attribute_names=[protected_attribute])

    rw = Reweighing(unprivileged_groups=[{protected_attribute: 0}], privileged_groups=[{protected_attribute: 1}])
    reweighed_dataset = rw.fit_transform(binary_label_dataset)

    X = reweighed_dataset.features
    y = reweighed_dataset.labels.ravel()
    weights = reweighed_dataset.instance_weights
    return X, y, weights

methods = {
    "Disparate Impact Remover": run_disparate_impact_remover,
    "Reweighing": run_reweighing
}

results = {}

# Train and evaluate models for each method and classifier
for method_name, method in methods.items():
    print(f"Processing: {method_name}")

    if method_name == "Disparate Impact Remover":
        X_transformed, y_transformed, weights = method(features, labels)
    else:
        X_transformed, y_transformed, weights = method(features, labels)

    if weights is not None:
        X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(X_transformed, y_transformed, weights, test_size=0.2, random_state=42)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X_transformed, y_transformed, test_size=0.2, random_state=42)
        w_train = None  # Setting weights to None for non-Reweighing methods

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
        "Support Vector Machine": SVC(probability=True, random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Gradient Boosting Machines": XGBClassifier(random_state=42)
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

        dataset_true = BinaryLabelDataset(df=pd.DataFrame(np.column_stack([X_test, y_test, X_test[protected_attribute]]),
                                                          columns=list(features.columns) + ['income', protected_attribute]),
                                          label_names=['income'],
                                          protected_attribute_names=[protected_attribute])

        dataset_pred = dataset_true.copy()
        dataset_pred.labels = predictions.reshape(-1, 1)

        metric = ClassificationMetric(dataset_true, dataset_pred,
                                      unprivileged_groups=[{protected_attribute: 0}],
                                      privileged_groups=[{protected_attribute: 1}])

        results[f"{method_name} - {model_name}"] = {
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

# Comparison of all methods
for name, result in results.items():
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

