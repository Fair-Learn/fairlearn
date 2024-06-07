from flask import Flask, render_template, redirect, url_for, request, session, send_file
# import matplotlib.pyplot as plt
# import io
# import base64
# import pandas as pd
# import seaborn as sns
import asyncio
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

    
    controller = Controller()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    imglist = loop.run_until_complete(controller.first_part_results_tables())




    return render_template('result.html', progress=100, dataset=selected_dataset, model=selected_model, test_size=test_size, random_state=random_state, privileged_groups=privileged_groups, imglist=imglist)

@app.route('/team')
def team():
    return render_template('team.html')

@app.route('/resource')
def resource():
    return render_template('resource.html')

if __name__ == '__main__':
    app.run(debug=True)
