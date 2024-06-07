import asyncio
from service.thirdpart import ThirdPart
# controller.py içeriği
import sys
import os

# `/service` dizinini `sys.path` içerisine ekleyin
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../service')))

from service.firstpart import FirstPart
# from service.secondPart import SecondPart
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

    async def third_part(self, test_size=0.3, random_state=82, privileged_group="telephone"):
        analyzer = ThirdPart(test_size, random_state, privileged_group)
        await analyzer.run_experiment()
        analyzer.display_results()
        

# aa = AsyncWorker()
# asyncio.run(aa.process_data())
