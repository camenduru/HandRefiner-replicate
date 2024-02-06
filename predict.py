import os
from cog import BasePredictor, Input, Path
import sys
sys.path.append('/content/HandRefiner')
os.chdir('/content/HandRefiner')
import subprocess

def inference(image, prompt, seed):
    command = f"python handrefiner.py --input_img {image} --out_dir /content/HandRefiner/output --strength 0.55 --weights /content/HandRefiner/models/inpaint_depth_control.ckpt --prompt '{prompt}' --seed {seed}"
    try:
        result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        output_path = '/content/HandRefiner/output/image_0.jpg'
        print("Output:", result.stdout)
        return output_path
    except subprocess.CalledProcessError as e:
        print("Error:", e.stderr)
        return None

class Predictor(BasePredictor):
    def setup(self) -> None:
        self.net=BriaRMBG()
        model_path = hf_hub_download("briaai/RMBG-1.4", 'model.pth')
        if torch.cuda.is_available():
            self.net.load_state_dict(torch.load(model_path))
            self.net=self.net.cuda()
        else:
            self.net.load_state_dict(torch.load(model_path,map_location="cpu"))
        self.net.eval()
    def predict(
        self,
        input_image: Path = Input(description="Input Image"),
        prompt: str = Input(default="a cute rabbit, white background, pastel hues, minimal illustration, line art, pen drawing"),
        seed: int = Input(default=34343),
    ) -> Path:
        output_image = inference(input_image, prompt, seed)
        return Path(output_image)