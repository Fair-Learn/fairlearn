from flask import Flask, render_template, redirect, url_for, request, session, send_file
import matplotlib.pyplot as plt
from math import pi
import io
import base64
import asyncio
import pandas as pd
from controller.controller import Controller

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
    group = None
    tsize = int(test_size)
    rstate = int(random_state)
    
    if privileged_groups == "1":
        group = "foreign_worker"
    else:
        group = "telephone"
    print(group)
    controller = Controller(selected_model, tsize/100, rstate, group)
    # controller = Controller(selected_model)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    modelimg = loop.run_until_complete(controller.first_part_results_tables())
    imglist2 = loop.run_until_complete(controller.second_part_results_tables())
    imglist3 = loop.run_until_complete(controller.third_part_results_tables())
    imm = controller.metotimg

    print("******************************************")
    print("******************************************")
    print("******************************************")


    for name, result in controller.allresults.items():
        print(name)

    print("******************************************")
    print("******************************************")
    print("******************************************")

    
    
    
    # categories = ['Disparate Impact', 'Statistical Parity Difference', 'Equal Opportunity Difference', 'Average Odds Difference', 'Theil Index']
    # num_vars = len(categories)

    # relabeller = controller.allresults["Relabeller"]
    # diremover = controller.allresults["Disparate Impact Remover"]
    # reweighing = controller.allresults["Reweighing"]
    # # Veri setlerini tanımlayın
    # values1 = [round(relabeller['disparate_impact'], 2), round(relabeller['statistical_parity_difference'], 2), round(relabeller['equal_opportunity_difference'], 2), round(relabeller['average_odds_difference'], 2), round(relabeller['theil_index'], 2)]
    # values2 = [round(diremover['disparate_impact'], 2), round(diremover['statistical_parity_difference'], 2), round(diremover['equal_opportunity_difference'], 2), round(diremover['average_odds_difference'], 2), round(diremover['theil_index'], 2)]
    # values3 = [round(reweighing['disparate_impact'], 2), round(reweighing['statistical_parity_difference'], 2), round(reweighing['equal_opportunity_difference'], 2), round(reweighing['average_odds_difference'], 2), round(reweighing['theil_index'], 2)]


    # values1 += values1[:1]
    # values2 += values2[:1]
    # values3 += values3[:1]

    # angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
    # angles += angles[:1]

    # fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    # ax.plot(angles, values1, linewidth=2, linestyle='solid', label='Relabeller')
    # ax.fill(angles, values1, 'b', alpha=0.4)

    # ax.plot(angles, values2, linewidth=2, linestyle='solid', label='Disparate Impact Remover')
    # ax.fill(angles, values2, 'r', alpha=0.4)

    # ax.plot(angles, values3, linewidth=2, linestyle='solid', label='Reweighing')
    # ax.fill(angles, values3, 'g', alpha=0.4)

    # ax.set_xticks(angles[:-1])
    # ax.set_xticklabels(categories)

    # ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

    # # Grafiği kaydetmek için
    # img = io.BytesIO()
    # plt.savefig(img, format='png')
    # img.seek(0)
    # radar = base64.b64encode(img.getvalue()).decode()


    radar = controller.radarchart()

    relab = controller.allresults["Relabeller"]
    report_df = pd.DataFrame(relab['report']).transpose()
    report_df = report_df.round(2)
    relabeller_report_html = report_df.to_html(classes='table table-striped')

    relab = controller.allresults["Reweighing"]
    report_df = pd.DataFrame(relab['report']).transpose()
    report_df = report_df.round(2)
    reweighing_report_html = report_df.to_html(classes='table table-striped')

    relab = controller.allresults["Disparate Impact Remover"]
    report_df = pd.DataFrame(relab['report']).transpose()
    report_df = report_df.round(2)
    dir_report_html = report_df.to_html(classes='table table-striped')

    controller.createchart()

    return render_template('result.html', progress=100, dataset=selected_dataset, model=selected_model, test_size=test_size, random_state=random_state, privileged_groups=privileged_groups, modelimg=modelimg, imm = imm, radar = radar, relabeller_report_html=relabeller_report_html, reweighing_report_html=reweighing_report_html, dir_report_html=dir_report_html, allresults=controller.allresults)

@app.route('/team')
def team():
    return render_template('team.html')

@app.route('/resource')
def resource():
    return render_template('resource.html')

if __name__ == '__main__':
    app.run(debug=True)
