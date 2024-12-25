from evaluator.evaluator import Evaluator
from dataset import Dataset

config = {}
config["save_dir"] = "/home/qhj/yiruo/evaluate/try_result"
config["save_metric_score"] = True
config["metrics"] = [ 'bleu-3']
config["save_intermediate_data"] = True
config["metric_setting"] = {"bleu_max_order": 1}


evaluator = Evaluator(config)

golden_file_path = "/home/qhj/yiruo/baselines/dataset/golden_response.json"
user_file_path = "/home/qhj/yiruo/baselines/output/deepseek/response.json"


sample_data = [
    {
        "sample_id": "1",
        "golden_response": ["Paris"],
        "predicted_response": "Paris",
    },
    {
        "sample_id": "2",
        "golden_response": ["George Orwell"],
        "predicted_response": "George Orwell",
    },
    {
        "sample_id": "3",
        "golden_response": ["In Paris"],
        "predicted_response": "In Paris",
    }
]

dataset = Dataset(data=sample_data)


print(evaluator.evaluate(dataset))


#evaluator.evaluate(data)