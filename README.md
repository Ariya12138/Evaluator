# Evaluator

复用了FlashRAG评测生成答案质量的Evaluator，做到了即插即用。
支持metrics: rouge-1,rouge-2, rouge-l, bleu-1, bleu-2, bleu-3, bleu-4, f1, recall, precision。（其他的还没测）


数据集格式如下：  
generated_response_file.jsonl
```
{“sample_id": $sample_id, "predicted_response": str}
```

golden_response_file.jsonl
```
{“sample_id": $sample_id, "golden_response": str}
```


如果有其他数据集格式，可以在dataset.py中的Dataset类的_load_data中重新实现如何读取数据
```
    def _load_data(self, golden_file_path: str, user_file_path: str) -> List[Item]:
        """Load data from the provided dataset_path or directly download the file(TODO)."""
        if not os.path.exists(golden_file_path):
            # TODO: auto download: self._download(self.dataset_name, dataset_path)
            raise FileNotFoundError(f"Dataset file {golden_file_path} not found.")
        
        if not os.path.exists(user_file_path):
            raise FileNotFoundError(f"Dataset file {user_file_path} not found.")

        data = []
        with open(golden_file_path, "r", encoding="utf-8") as fg, open(user_file_path, "r", encoding="utf-8") as fu:
            for line1, line2 in zip(fg, fu):
                golden_dict = json.loads(line1)
                user_dict = json.loads(line2)
                assert golden_dict["sample_id"] == user_dict["sample_id"]
                
                if user_dict["predicted_response"] == None:
                    continue
                item_dict = {}
                item_dict["sample_id"] = golden_dict["sample_id"]
                temp_user_response = process_response(user_dict["predicted_response"])
                temp_golden_response = process_response(golden_dict["golden_response"])
                if temp_user_response == None or temp_golden_response == None:
                    continue
                item_dict["golden_response"] = [temp_golden_response]
                item_dict["predicted_response"] = temp_user_response
                item = Item(item_dict)
                data.append(item)
        

        return data
```
