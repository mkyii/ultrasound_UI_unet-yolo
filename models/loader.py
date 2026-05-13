import torch
import segmentation_models_pytorch as smp


def build_model(device):
    """
    EfficientNet-B1 encoder 기반 UNet segmentation model 생성.
    """
    model = smp.Unet(
        encoder_name="efficientnet-b1",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation=None,
    )

    model = model.to(device)

    return model


def load_model(path, device):
    """
    학습된 checkpoint를 로드하고 evaluation mode로 전환.
    """
    model = build_model(device)

    state_dict = torch.load(
        path,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()

    return model