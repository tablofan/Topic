# import pytorch_lightning as pl
import copy
from src.ml.resnet import BasicBlock, Resnet
import torch
import torch.nn as nn
from loguru import logger

if __name__ == "__main__":

    # Use our own
    resnet18 = Resnet(num_classes=10, block=BasicBlock, layers=[2, 2, 2, 2])

    example_image = torch.rand(1, 3, 256, 256)  # [B, C, H, W]

    print(resnet18(example_image).shape)

    # Use pretrained from https://pytorch.org/vision/stable/models.html
    torchvision_resnet18 = torch.hub.load(
        "pytorch/vision", "resnet50", weights="IMAGENET1K_V2"
    )

    # How to replace a layer
    def _surgeon_resnet_first_layer(
        model: nn.Module, new_in_features: int, new_num_classes: int
    ) -> nn.Module:
        new_model = copy.deepcopy(model)

        new_model.conv1 = nn.Conv2d(
            new_in_features,
            model.conv1.out_channels,
            kernel_size=model.conv1.kernel_size,
            stride=model.conv1.stride,
            padding=model.conv1.padding,
            bias=model.conv1.bias,
        )
        
        new_model.fc = nn.Linear(
            in_features=model.fc.in_features, out_features=new_num_classes, bias=True if not isinstance(type(model.fc.bias), type(None)) else False
        )

        return new_model

    new_num_classes = 10
    new_resnet18 = _surgeon_resnet_first_layer(
        model=torchvision_resnet18, new_in_features=1, new_num_classes=new_num_classes
    )

    assert new_resnet18 != torchvision_resnet18

    print(new_resnet18)


    # proof we changed the first layer
    example_image = torch.rand(1, 3, 256, 256)  # [B, C, H, W]
    e = None
    try:
        new_resnet18(example_image)

    except RuntimeError as e:
        logger.error(e)


    # proof we changed the last layer
    example_image = torch.rand(1, 1, 256, 256)
    output = new_resnet18(example_image)
    assert output.shape[-1] == new_num_classes; logger.success(f"{output.shape[-1]=} == {new_num_classes=}")

    


