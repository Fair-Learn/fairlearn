import asyncio
from service.thirdpart import ThirdPart
import sys
import os
import io
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import base64


from service.firstpart import FirstPart
from service.secondpart import SecondPart
from service.thirdpart import ThirdPart


class Controller:
    def __init__(self):
        self.num = None

    async def process_data(self):
        experiment = ThirdPart()
        await experiment.run_experiment()
        experiment.display_results()

    async def first_part(self, test_size=0.3, random_state=82, privileged_group="telephone"):
        analyzer = FirstPart(test_size, random_state, privileged_group)
        return await analyzer.loop()

    async def second_part(self, test_size=0.3, random_state=82, privileged_group="telephone"):
        analyzer = SecondPart(test_size, random_state, privileged_group)
        return await analyzer.run_all_methods()

    async def third_part(self, test_size=0.3, random_state=82, privileged_group="telephone"):
        analyzer = ThirdPart(test_size, random_state, privileged_group)
        await analyzer.run_experiment()
        analyzer.display_results()

    async def first_part_results_tables(self, test_size=0.3, random_state=82, privileged_group="telephone"):
        result = await self.first_part(test_size, random_state, privileged_group)
        imglist = []
        for name, result in result.items():
            bimg = io.BytesIO()
            plt.figure(figsize=(8, 6))
            sns.heatmap(result['conf_matrix'], annot=True, fmt="d", cmap="Blues", xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
            plt.title(f'Confusion Matrix - {name}')
            plt.xlabel('Predicted label')
            plt.ylabel('True label')
            plt.savefig(bimg, format='png')
            bimg.seek(0)
            bplot_url = base64.b64encode(bimg.getvalue()).decode()
            imglist.append(bplot_url)
        
        return imglist

    async def second_part_results_tables(self, test_size=0.3, random_state=82, privileged_group="telephone"):
        result = await self.second_part(test_size, random_state, privileged_group)
        imglist = []
        for name, result in result.items():
            bimg = io.BytesIO()
            plt.figure(figsize=(4, 3))
            sns.heatmap(result['conf_matrix'], annot=True, fmt="d", cmap="Blues", xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
            plt.title(f'Confusion Matrix - {name}')
            plt.xlabel('Predicted label')
            plt.ylabel('True label')
            plt.savefig(bimg, format='png')
            bimg.seek(0)
            bplot_url = base64.b64encode(bimg.getvalue()).decode()
            imglist.append(bplot_url)
        
        return imglist

# aa = AsyncWorker()
# asyncio.run(aa.process_data())
