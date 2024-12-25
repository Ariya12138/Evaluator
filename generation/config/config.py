import re
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import yaml
import random
import datetime


class Config:
    def __init__(self, config_file_path=None, config_dict={}):

        self.yaml_loader = self._build_yaml_loader()
        self.file_config = self._load_file_config(config_file_path)
        self.variable_config = config_dict

        self.external_config = self._merge_external_config()

        self.internal_config = self._get_internal_config()

        self.final_config = self._get_final_config()

        self._check_final_config()
        self._set_additional_key()

        self._init_device()
        self._prepare_dir()

    def _build_yaml_loader(self):
        loader = yaml.FullLoader
        loader.add_implicit_resolver(
            "tag:yaml.org,2002:float",
            re.compile(
                """^(?:
             [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
            |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
            |\\.[0-9_]+(?:[eE][-+][0-9]+)?
            |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
            |[-+]?\\.(?:inf|Inf|INF)
            |\\.(?:nan|NaN|NAN))$""",
                re.X,
            ),
            list("-+0123456789."),
        )
        return loader

    def _load_file_config(self, config_file_path: str):
        file_config = dict()
        if config_file_path:
            with open(config_file_path, "r", encoding="utf-8") as f:
                file_config.update(yaml.load(f.read(), Loader=self.yaml_loader))
        return file_config

    @staticmethod
    def _update_dict(old_dict: dict, new_dict: dict):
        # Update the original update method of the dictionary:
        # If there is the same key in `old_dict` and `new_dict`, and value is of type dict, update the key in dict

        same_keys = []
        for key, value in new_dict.items():
            if key in old_dict and isinstance(value, dict):
                same_keys.append(key)
        for key in same_keys:
            old_item = old_dict[key]
            new_item = new_dict[key]
            old_item.update(new_item)
            new_dict[key] = old_item

        old_dict.update(new_dict)
        return old_dict

    def _merge_external_config(self):
        external_config = dict()
        external_config = self._update_dict(external_config, self.file_config)
        external_config = self._update_dict(external_config, self.variable_config)

        return external_config

    def _get_internal_config(self):
        current_path = os.path.dirname(os.path.realpath(__file__))
        init_config_path = os.path.join(current_path, "basic_config.yaml")
        if os.path.exists(init_config_path):
            internal_config = self._load_file_config(init_config_path)
        else:
            internal_config = dict()

        return internal_config

    def _get_final_config(self):
        final_config = dict()
        final_config = self._update_dict(final_config, self.internal_config)
        final_config = self._update_dict(final_config, self.external_config)

        return final_config

    def _check_final_config(self):
        # check split
        split = self.final_config.get("split")
        if split is None:
            split = ["train", "dev", "test"]
        if isinstance(split, str):
            split = [split]
        self.final_config["split"] = split

    def _init_device(self):
        gpu_id = self.final_config["gpu_id"]
        if gpu_id is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        try:
            # import pynvml 
            # pynvml.nvmlInit()
            # gpu_num = pynvml.nvmlDeviceGetCount()
            import torch
            gpu_num = torch.cuda.device_count()
        except:
            gpu_num = 0
        self.final_config['gpu_num'] = gpu_num
        if gpu_num > 0:
            self.final_config["device"] = "cuda"
        else:
            self.final_config['device'] = 'cpu'

    
    def _set_additional_key(self):
        data_dir = self.final_config["user_data_dir"]
        self.final_config["user_dataset_path"] = os.path.join(data_dir, "response.json")
        

    def _prepare_dir(self):
        save_note = self.final_config["save_note"]
        current_time = datetime.datetime.now()
        self.final_config["save_dir"] = os.path.join(
            self.final_config["save_dir"],
            f"{self.final_config['dataset_name']}_{self.final_config['method_name']}_{self.final_config['model_name']}_{self.final_config['passage_type']}_{current_time.strftime('%Y_%m_%d_%H_%M')}_{save_note}",
        )
        os.makedirs(self.final_config["save_dir"], exist_ok=True)
        # save config parameters
        config_save_path = os.path.join(self.final_config["save_dir"], "config.yaml")
        with open(config_save_path, "w") as f:
            yaml.dump(self.final_config, f)

   

    def __setitem__(self, key, value):
        if not isinstance(key, str):
            raise TypeError("index must be a str.")
        self.final_config[key] = value

    def __getattr__(self, item):
        if "final_config" not in self.__dict__:
            raise AttributeError("'Config' object has no attribute 'final_config'")
        if item in self.final_config:
            return self.final_config[item]
        raise AttributeError(f"'Config' object has no attribute '{item}'")

    def __getitem__(self, item):
        return self.final_config.get(item)

    def __contains__(self, key):
        if not isinstance(key, str):
            raise TypeError("index must be a str.")
        return key in self.final_config

    def __repr__(self):
        return self.final_config.__str__()
