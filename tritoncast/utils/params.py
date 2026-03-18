# coding: utf-8

import logging
from ruamel.yaml import YAML


class YParams:
    """Yaml file parser shared across experiments."""

    def __init__(self, yaml_filename, config_name, print_params=False):
        self._yaml_filename = yaml_filename
        self._config_name = config_name
        self.params = {}

        if print_params:
            print("------------------ Configuration ------------------ ", yaml_filename)

        with open(yaml_filename, "rb") as yaml_file:
            yaml = YAML().load(yaml_file)
            for key, val in yaml[config_name].items():
                if print_params:
                    print(key, val)
                if val == "None":
                    val = None

                self.params[key] = val
                setattr(self, key, val)

        if print_params:
            print("---------------------------------------------------")

    def __getitem__(self, key):
        return self.params[key]

    def __setitem__(self, key, val):
        self.params[key] = val
        setattr(self, key, val)

    def __contains__(self, key):
        return key in self.params

    def update_params(self, config):
        for key, val in config.items():
            self.params[key] = val
            setattr(self, key, val)

    def log(self):
        logging.info("------------------ Configuration ------------------")
        logging.info("Configuration file: %s", self._yaml_filename)
        logging.info("Configuration name: %s", self._config_name)
        for key, val in self.params.items():
            logging.info("%s %s", key, val)
        logging.info("---------------------------------------------------")

