import asyncio
from service.thirdpart import ThirdPart
import sys
import os
import io
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import base64
import numpy as np
import matplotlib.pyplot as plt
from math import pi


from service.firstpart import FirstPart
from service.secondpart import SecondPart
from service.thirdpart import ThirdPart


class Controller:
    def __init__(self, model = "Logistic Regression", test_size=0.3, random_state=82, privileged_group="telephone"):
        self.num = None
        self.firstpart = None
        self.secondpart = None
        self.thirdpart = None
        self.model = model
        self.test_size = test_size
        self.random_state = random_state
        self.privileged_group = privileged_group
        self.allresults = {}
        self.allresults["model"] = model
        self.takepart()
        self.modelimg = None
        self.metotimg = []

    async def process_data(self):
        experiment = ThirdPart()
        await experiment.run_experiment()
        experiment.display_results()

    async def first_part(self):
        analyzer = FirstPart(self.test_size, self.random_state, self.privileged_group)
        self.first_part = await analyzer.loop()
        return self.first_part

    async def second_part(self):
        analyzer = SecondPart(self.test_size, self.random_state, self.privileged_group)
        self.second_part = await analyzer.run_all_methods()
        return self.second_part

    async def third_part(self):
        analyzer = ThirdPart(self.test_size, self.random_state, self.privileged_group)
        self.third_part = await analyzer.run_experiment()
        # analyzer.display_results()
        return self.third_part

    async def first_part_results_tables(self):
        result = await self.first_part()
        img = None
        for name, result in result.items():
            if self.model in name:
                self.allresults["modeldata"] = result
                bimg = io.BytesIO()
                plt.figure(figsize=(5, 5))
                sns.heatmap(result['conf_matrix'], annot=True, fmt="d", cmap="Blues", xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
                plt.title(f'{name}')
                plt.xlabel('Predicted label')
                plt.ylabel('True label')
                plt.savefig(bimg, format='png')
                bimg.seek(0)
                bplot_url = base64.b64encode(bimg.getvalue()).decode()
                img = bplot_url
        
        return img

    async def second_part_results_tables(self):
        result = await self.second_part()
        imglist = []
        for name, result in result.items():
            if self.model in name:
                methodname = self.get_first_part(name)
                self.allresults[methodname] = result
                aa = methodname + "chart"
                bimg = io.BytesIO()
                plt.figure(figsize=(5, 5))
                sns.heatmap(result['conf_matrix'], annot=True, fmt="d", cmap="Blues", xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
                plt.title(f'{name}')
                plt.xlabel('Predicted label')
                plt.ylabel('True label')
                plt.savefig(bimg, format='png')
                bimg.seek(0)
                bplot_url = base64.b64encode(bimg.getvalue()).decode()
                imglist.append(bplot_url)
                self.allresults[aa] = bplot_url
                self.metotimg.append(bplot_url)
        
        return imglist

    async def third_part_results_tables(self):
        result = await self.third_part()
        imglist = []
        for name, result in result.items():
            if self.model in name:            
                methodname = self.get_first_part(name)
                self.allresults[methodname] = result
                aa = methodname + "chart"
                bimg = io.BytesIO()
                plt.figure(figsize=(5, 5))
                sns.heatmap(result['conf_matrix'], annot=True, fmt="d", cmap="Blues", xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
                plt.title(f'{name}')
                plt.xlabel('Predicted label')
                plt.ylabel('True label')
                plt.savefig(bimg, format='png')
                bimg.seek(0)
                bplot_url = base64.b64encode(bimg.getvalue()).decode()
                imglist.append(bplot_url)
                self.allresults[aa] = bplot_url
                self.metotimg.append(bplot_url)
        
        return imglist

    def takepart(self):
        if(self.model == "Logistic Regression"):
            self.model = "Logistic Regression"
        elif(self.model == "Support Vector Machine"):
            self.model = "SVC"
        elif(self.model == "Random Forest"):
            self.model = "Random Forest"
        elif(self.model == "Gradient Boosting Machines"):
            self.model = "XGBClassifier"

    def get_first_part(self, input_string):
        parts = input_string.split(' - ')
        return parts[0]


    def radarchart(self):
        # Kategorileri tanımlayın
        categories = ['Disparate Impact', 'Statistical Parity Difference', 'Equal Opportunity Difference', 'Average Odds Difference', 'Theil Index']
        num_vars = len(categories)

        relabeller = self.allresults["Relabeller"]
        diremover = self.allresults["Disparate Impact Remover"]
        reweighing = self.allresults["Reweighing"]
        # Veri setlerini tanımlayın
        values1 = [round(relabeller['disparate_impact'], 2), round(relabeller['statistical_parity_difference'], 2), round(relabeller['equal_opportunity_difference'], 2), round(relabeller['average_odds_difference'], 2), round(relabeller['theil_index'], 2)]
        values2 = [round(diremover['disparate_impact'], 2), round(diremover['statistical_parity_difference'], 2), round(diremover['equal_opportunity_difference'], 2), round(diremover['average_odds_difference'], 2), round(diremover['theil_index'], 2)]
        values3 = [round(reweighing['disparate_impact'], 2), round(reweighing['statistical_parity_difference'], 2), round(reweighing['equal_opportunity_difference'], 2), round(reweighing['average_odds_difference'], 2), round(reweighing['theil_index'], 2)]

        # İlk ve son değerin aynı olması için her veri setine ilk değeri ekleyin
        values1 += values1[:1]
        values2 += values2[:1]
        values3 += values3[:1]

        angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))

        ax.plot(angles, values1, linewidth=2, linestyle='solid', label='Relabeller')
        ax.fill(angles, values1, 'b', alpha=0.4)

        ax.plot(angles, values2, linewidth=2, linestyle='solid', label='Disparate Impact Remover')
        ax.fill(angles, values2, 'r', alpha=0.4)

        ax.plot(angles, values3, linewidth=2, linestyle='solid', label='Reweighing')
        ax.fill(angles, values3, 'g', alpha=0.4)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)

        ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

        # Grafiği kaydetmek için
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        chart = base64.b64encode(img.getvalue()).decode()


        return chart
        
    
    def createchart(self):
        

        relabeller = self.allresults["Relabeller"]
        metot_values = [relabeller["disparate_impact"], relabeller["statistical_parity_difference"], relabeller["equal_opportunity_difference"], relabeller["average_odds_difference"], relabeller["theil_index"]]
        self.allresults["RelabellerChart"] = self.makechart(metot_values)

        diremover = self.allresults["Disparate Impact Remover"]
        metot_values = [diremover["disparate_impact"], diremover["statistical_parity_difference"], diremover["equal_opportunity_difference"], diremover["average_odds_difference"], diremover["theil_index"]]
        self.allresults["DIRChart"] = self.makechart(metot_values)

        reweighing = self.allresults["Reweighing"]
        metot_values = [reweighing["disparate_impact"], reweighing["statistical_parity_difference"], reweighing["equal_opportunity_difference"], reweighing["average_odds_difference"], reweighing["theil_index"]]
        self.allresults["ReweighingChart"] = self.makechart(metot_values)
    
    
    def makechart(self, metot_values):
        labels = ['Disparate Impact', 'Statistical Parity Difference', 'Equal Opportunity Difference', 'Average Odds Difference', 'Theil Index']
        
        model = self.allresults["modeldata"]
        model_values = [model["disparate_impact"], model["statistical_parity_difference"], model["equal_opportunity_difference"], model["average_odds_difference"], model["theil_index"]]

        x = np.arange(len(labels))  # etiketlerin konumları
        width = 0.35  # çubuk genişliği

        fig, ax = plt.subplots(figsize=(10, 6))  # Grafik boyutunu ayarlama
        rects1 = ax.bar(x - width/2, model_values, width, label='Model', color='navy')
        rects2 = ax.bar(x + width/2, metot_values, width, label='Metot', color='lightgreen')

        # Eksen etiketleri ekleme
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Scores')
        ax.set_title('Comparison of Model and Metot')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()

        fig.tight_layout()

        # Grafiği belleğe kaydet
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        
        # Base64 kodlama
        chart = base64.b64encode(img.getvalue()).decode()

        return chart
        # aa = AsyncWorker()
        # asyncio.run(aa.process_data())
