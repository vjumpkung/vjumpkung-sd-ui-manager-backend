import os

class Envs:
    def __init__(self):
        self.CIVITAI_TOKEN = ""
        self.HUGGINGFACE_TOKEN = ""

    def get_enviroment_variable(self):
        if "CIVITAI_TOKEN" in os.environ.keys() and self.CIVITAI_TOKEN == "":
            self.CIVITAI_TOKEN = os.environ["CIVITAI_TOKEN"]
        if "HUGGINGFACE_TOKEN" in os.environ.keys() and self.HUGGINGFACE_TOKEN == "":
            self.HUGGINGFACE_TOKEN = os.environ["HUGGINGFACE_TOKEN"]

envs = Envs()