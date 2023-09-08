import numpy as np
import mindspore as ms

from mindcv.models import create_model


if __name__ == "__main__":
    ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend")

    model_name = "vit_b_32_224"
    network = create_model(model_name=model_name, pretrained=True)
    network.set_train(False)

    input_data = ms.Tensor(np.zeros([1, 3, 224, 224]), ms.float32)
    ms.export(network, input_data, file_name=f"./resources/{model_name}.mindir", file_format="MINDIR")
