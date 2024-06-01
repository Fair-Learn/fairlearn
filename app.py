from flask import Flask, render_template, redirect, url_for, request, session, send_file
import matplotlib.pyplot as plt
import io
import base64
from firstpart import FairnessAnalyzer
import pandas as pd
import seaborn as sns

app = Flask(__name__)
app.secret_key = 'your_secret_key'

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/demo')
def demo():
    return redirect(url_for('data'))

@app.route('/demo/data', methods=['GET', 'POST'])
def data():
    if request.method == 'POST':
        session['dataset'] = request.form['dataset']
        return redirect(url_for('model'))
    return render_template('data.html', progress=25)

@app.route('/demo/model', methods=['GET', 'POST'])
def model():
    selected_dataset = session.get('dataset', 'None')
    if request.method == 'POST':
        session['model'] = request.form['model']
        return redirect(url_for('parameter'))
    return render_template('model.html', progress=50, dataset=selected_dataset)

@app.route('/demo/parameter', methods=['GET', 'POST'])
def parameter():
    selected_dataset = session.get('dataset', 'None')
    selected_model = session.get('model', 'None')
    if request.method == 'POST':
        session['test_size'] = request.form['test_size']
        session['random_state'] = request.form['random_state']
        session['privileged_groups'] = request.form['privileged_groups']
        return redirect(url_for('result'))
    return render_template('parameter.html', progress=75, dataset=selected_dataset, model=selected_model)

@app.route('/demo/result')
def result():
    selected_dataset = session.get('dataset', 'None')
    selected_model = session.get('model', 'None')
    test_size = session.get('test_size', 'None')
    random_state = session.get('random_state', 'None')
    privileged_groups = session.get('privileged_groups', 'None')

    fair = FairnessAnalyzer(0.2, 82, "telephone")
    results = fair.loop()

    plotlist = []
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
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_urli = base64.b64encode(img.getvalue()).decode()
        plotlist.append(plot_urli)
        # plt.show()

    # Verilerle grafiği oluşturma
    values = [0.123, 0.258, 0.879]
    img = io.BytesIO()
    plt.figure(figsize=(10, 5))
    plt.plot(values, marker='o')
    plt.title('Example Data Plot')
    plt.xlabel('Index')
    plt.ylabel('Values')
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()



    return render_template('result.html', progress=100, dataset=selected_dataset, model=selected_model, test_size=test_size, random_state=random_state, privileged_groups=privileged_groups, plot_url=plot_url)

@app.route('/team')
def team():
    return render_template('team.html')

@app.route('/resource')
def resource():
    return render_template('resource.html')

if __name__ == '__main__':
    app.run(debug=True)
