import os


class Envs:
    def __init__(self):
        self.CIVITAI_TOKEN = ""
        self.HUGGINGFACE_TOKEN = ""

    def set_huggingface_token(self, value: str) -> None:
        """Keep the backend token and Hugging Face CLI environment in sync."""
        self.HUGGINGFACE_TOKEN = value
        os.environ["HUGGINGFACE_TOKEN"] = value
        os.environ["HF_TOKEN"] = value

    def get_environment_variable(self) -> None:
        if "CIVITAI_TOKEN" in os.environ and self.CIVITAI_TOKEN == "":
            self.CIVITAI_TOKEN = os.environ["CIVITAI_TOKEN"]

        if self.HUGGINGFACE_TOKEN == "":
            token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
            if token:
                self.set_huggingface_token(token)

    def get_enviroment_variable(self) -> None:
        """Backward-compatible alias for the original misspelled method name."""
        self.get_environment_variable()


envs = Envs()
