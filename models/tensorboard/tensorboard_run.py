from utils import get_project_root
import os
from tensorboard import program

tracking_address = os.path.join(
    get_project_root(),
    "models",
    "tensorboard",
    "fit",
    "20220523-112855"
)

if __name__ == "__main__":
    print(tracking_address)
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', tracking_address])
    url = tb.launch()
    print(f"Tensorflow listening on {url}")
