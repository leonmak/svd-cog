# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path
from .workflow_api import run

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        # self.model = torch.load("./weights.pth")

    def predict(
        self,
        image_text: str = Input(description="Input image text"),
        # image: Path = Input(description="Input image"),
        # scale: float = Input(
        #     description="Factor to scale image by", ge=0, le=10, default=1.5
        # ),
    ) -> Path:
        """Run a single prediction on the model"""
        # image_path_str = image.absolute().as_posix() if image else ''
        return run(image_text)
        # processed_input = preprocess(image)
        # output = self.model(processed_image, scale)
        # return postprocess(output)
