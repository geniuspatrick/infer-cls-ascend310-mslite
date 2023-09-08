# Inference of Classification Model on Ascend310 via MindSpore/Mindspore-lite

## Before Starting

### 0. Setup Env

- Get Ascend310 processor.
- Install Ascend toolkit software.
- Install MindSpore or MindSpore-lite.

### 1. Export MindIR

If you get a trained model on mindspore, by [`mindcv`](https://github.com/mindspore-lab/mindcv) for example, export it to `mindir` format by:

```python
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
```

The above code snippet export `ViT` to `mindir` format with `batch_size=1`.
Or you can simply run:

```shell
python ./scripts/export.py
```

##  One Click Run

We provide a one-click-run shell script to easily finish all the jobs like building, inferring, calculating, etc.
Just run:

```shell
./scripts/run.sh ./resources/model.mindir /path/to/imagenet/val [device_id]
```

where `device_id` is optional.

## Step by Step

### 0. Compile

```shell
MINDSPORE_PATH="`pip show mindspore-ascend | grep Location | awk '{print $2"/mindspore"}' | xargs realpath`"
if [[ ! $MINDSPORE_PATH ]];then
    MINDSPORE_PATH="`pip show mindspore | grep Location | awk '{print $2"/mindspore"}' | xargs realpath`"
fi
fi
mkdir -p build && cd build
cmake .. -DMINDSPORE_PATH=$MINDSPORE_PATH
make -j8
cd ..
```

### 1. Inference

```shell
if [ -d outputs ]; then
    rm -rf outputs
fi
mkdir -p outputs
./build/main \
    --mindir_path=./resources/model.mindir \
    --dataset_path=/path/to/imagenet/val \
    --output_path=outputs \
    --device_type=Ascend \
    --device_id=0
```

### 2. Calculate

```shell
python ./scripts/postprocess.py \
    --dataset_path=/path/to/imagenet/val \
    --results_path=outputs
```

## TODOs

- [ ] MindSpore-lite
