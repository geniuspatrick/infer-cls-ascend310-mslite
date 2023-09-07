"""postprocess for 310 inference"""

import os
import argparse
import numpy as np


def cal_acc_imagenet(dataset_path, results_path):
    labels = {}
    for idx, subdir in enumerate(sorted(os.listdir(dataset_path))):
        for filename in sorted(os.listdir(os.path.join(dataset_path, subdir))):
            img_name = os.path.splitext(filename)[0]
            labels[img_name] = idx
    print(f"The number of images: {len(labels)}")

    top1 = 0
    top5 = 0
    batch_size = 1
    result_shape = (batch_size, 1000)
    result_files = os.listdir(results_path)
    n_results = len(result_files)
    print(f"The number of bin results: {n_results}")
    for result_file in result_files:
        img_name = result_file.split('_0.')[0]
        logits = np.fromfile(os.path.join(results_path, result_file), dtype=np.float32).reshape(result_shape)
        for batch in range(batch_size):
            predict = np.argsort(-logits[batch], axis=-1)
            if labels[img_name] == predict[0]:
                top1 += 1
            if labels[img_name] in predict[:5]:
                top5 += 1
    print(f"Top1 acc: {top1 / n_results}")
    print(f"Top5 acc: {top5 / n_results}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="resnet inference")
    parser.add_argument("--dataset_path", type=str, required=True, help="path to IN1K val subset.")
    parser.add_argument("--results_path", type=str, required=True, help="path to dumped bin results.")
    args = parser.parse_args()

    cal_acc_imagenet(args.dataset_path, args.results_path)
