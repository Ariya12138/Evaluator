import os
import json
import random
import warnings
from typing import List, Dict, Any, Optional, Union
import numpy as np
from evaluator.utils import process_response


def convert_numpy(obj: Union[Dict, list, np.ndarray, np.generic]) -> Any:
    """Recursively convert numpy objects in nested dictionaries or lists to native Python types."""
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(i) for i in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert numpy arrays to lists
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()  # Convert numpy scalars to native Python scalars
    elif isinstance(obj, np.float32):
        return float(obj)
    else:
        return obj  # Return the object as-is if it's neither a dict, list, nor numpy type


class Item:
    """A container class used to store and manipulate a sample within a dataset.
    Information related to this sample during training/inference will be stored in `self.output`.
    Each attribute of this class can be used like a dict key (also for key in `self.output`).
    """

    def __init__(self, item_dict: Dict[str, Any]) -> None:
        self.sample_id: Optional[str] = item_dict.get("sample_id", None)
        self.golden_answers: List[str] = item_dict.get("golden_response", [])
        self.pred: List[str] = item_dict.get("predicted_response", [])
        self.output: Dict[str, Any] = item_dict.get("output", {})
        self.data: Dict[str, Any] = item_dict

    def update_output(self, key: str, value: Any) -> None:
        """Update the output dict and keep a key in self.output can be used as an attribute."""
        if key in ["id", "question", "golden_answers", "output", "choices"]:
            raise AttributeError(f"{key} should not be changed")
        else:
            self.output[key] = value

    def update_evaluation_score(self, metric_name: str, metric_score: float) -> None:
        """Update the evaluation score of this sample for a metric."""
        if "metric_score" not in self.output:
            self.output["metric_score"] = {}
        self.output["metric_score"][metric_name] = metric_score

    def __getattr__(self, attr_name: str) -> Any:
        predefined_attrs = ["sample_id", "golden_answers","pred"]
        if attr_name in predefined_attrs:
            return super().__getattribute__(attr_name)
        else:
            output = self.output
            if attr_name in output:
                return output[attr_name]
            else:
                try:
                    return self.data[attr_name]
                except AttributeError:
                    raise AttributeError(f"Attribute `{attr_name}` not found")

    def to_dict(self) -> Dict[str, Any]:
        """Convert all information within the data sample into a dict. Information generated
        during the inference will be saved into output field.
        """


        output = {
            "sample_id": self.sample_id,
            "golden_answers": self.golden_answers,
            "predicted_response": self.pred,
            "output": convert_numpy(self.output),
        }


        return output

    def __str__(self) -> str:
        """Return a string representation of the item with its main attributes."""
        return json.dumps(self.to_dict(), indent=4)


class Dataset:
    """A container class used to store the whole dataset. Inside the class, each data sample will be stored
    in `Item` class. The properties of the dataset represent the list of attributes corresponding to each item in the dataset.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        data: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        
        if data is None:
            if config is not None:
                self.config = config
            golden_file_path = config.get("golden_dataset_path", None)
            user_file_path = config.get("user_dataset_path", None)
            if golden_file_path is None or user_file_path is None:
                raise ValueError("Please provide the golden and user dataset path.")
            else:
                self.data = self._load_data(golden_file_path, user_file_path) 
        else:     
            if isinstance(data[0], dict):
                self.data = [Item(item_dict) for item_dict in data]
            else:
                assert isinstance(data[0], Item)
                self.data = data

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
    def update_output(self, key: str, value_list: List[Any]) -> None:
        """Update the overall output field for each sample in the dataset."""
        assert len(self.data) == len(value_list)
        for item, value in zip(self.data, value_list):
            item.update_output(key, value)

    @property
    def pred(self) -> List[Optional[str]]:
        return [item.pred for item in self.data]

    @property
    def golden_answers(self) -> List[List[str]]:
        return [item.golden_answers for item in self.data]

    @property
    def sample_id(self) -> List[Optional[str]]:
        return [item.sample_id for item in self.data]



    def __getattr__(self, attr_name: str) -> List[Any]:
        return [item.__getattr__(attr_name) for item in self.data]

    def get_attr_data(self, attr_name: str) -> List[Any]:
        """For the attributes constructed later (not implemented using property),
        obtain a list of this attribute in the entire dataset.
        """
        return [item[attr_name] for item in self.data]

    def __getitem__(self, index: int) -> Item:
        return self.data[index]

    def __len__(self) -> int:
        return len(self.data)

    def save(self, save_path: str) -> None:
        """Save the dataset into the original format."""

        save_data = [item.to_dict() for item in self.data]
        def custom_serializer(obj):
            if isinstance(obj, np.float32):  
                return float(obj)      
            if isinstance(obj, np.bool_):
                return str(obj)     
            raise TypeError(f"Type {type(obj)} not serializable")
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(save_data, f, indent=4, default=custom_serializer)

    def __str__(self) -> str:
        """Return a string representation of the dataset with a summary of items."""
        return f"Dataset '{self.dataset_name}' with {len(self)} items"
