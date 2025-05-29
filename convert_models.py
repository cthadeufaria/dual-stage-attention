import torch
import os
from glob import glob
from dual_attention import DualAttention
from dataset import VideoDataset


def main():
    """
    Main function to convert PyTorch state dict model to TorchScript.
    Considerations about tracing and scripting: https://ppwwyyxx.com/blog/2022/TorchScript-Tracing-vs-Scripting/
    """
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Device set to: {device}")

    dataset = VideoDataset('./datasets/LIVE_NFLX_Plus')

    with torch.no_grad():
        example_input = (
            (
                (
                    torch.randn(3, 80, 224, 224).to(device), 
                    torch.randn(3, 320, 224, 224).to(device)
                ), 
                torch.randn(3, 10, 1080, 1920).to(device)
            ), 
            torch.randn(10, 4).to(device)
        )

    models_paths = glob('./runs/models/state_dict/*')
    model = DualAttention(device, dataset.max_duration)
    model.eval()

    for model_path in models_paths:
        model_name = model_path.split('/')[-1]

        if os.path.exists('./runs/models/torchscript/' + model_name + '.pt'):
            print("Model already converted:", model_name)
            continue

        print("Converting model:", model_name)

        try:
            model.load_state_dict(torch.load(model_path, weights_only=True))

        except RuntimeError as e:
            print("Error loading state dict:", e)
            print("Model not loaded. Skipping conversion.")
            continue

        traced_model = torch.jit.trace(model, (example_input,))
        traced_model.save('./runs/models/torchscript/' + model_name + '.pt')

        print("Model converted to TorchScript and saved as", model_name + '.pt')


if __name__ == "__main__":
    main()