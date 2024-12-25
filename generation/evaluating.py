import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from evaluator.evaluator import Evaluator
from config.config import Config
from dataset import Dataset



if __name__ == '__main__':
    
    my_config = Config(config_file_path = 'my_config.yaml')

    evaluator = Evaluator(my_config.final_config)
    data = Dataset(config = my_config.final_config)
    evaluator.evaluate(data)
    
    logger.info("Calculate Finish!")
