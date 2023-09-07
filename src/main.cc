#include <sys/time.h>
#include <dirent.h>
#include <iostream>
#include <string>
#include <algorithm>
#include <iosfwd>
#include <vector>

#include "infer.h"

DEFINE_string(mindir_path, "", "/path/to/mindir");
DEFINE_string(dataset_path, "", "/path/to/dataset");
DEFINE_string(output_path, "outputs", "/path/to/outputs");
DEFINE_string(device_type, "CPU", "device type");
DEFINE_int32(device_id, 0, "device id");

int main(int argc, char **argv) {
  if (!ParseCommandLineFlags(argc, argv)) {
    std::cout << "[ERROR] Failed to parse args" << std::endl;
    return 1;
  }

  Model model;
  if (!LoadModel(FLAGS_mindir_path, FLAGS_device_type, FLAGS_device_id, &model)) {
    std::cout << "[ERROR] Failed to load model " << FLAGS_mindir_path << ", device id: " << FLAGS_device_id
              << ", device type: " << FLAGS_device_type;
    return 1;
  }

  auto image_files = GetAllFiles(FLAGS_dataset_path);
  if (image_files.empty()) {
    std::cout << "[ERROR] Failed to get image file lists" << std::endl;
    return 1;
  }
  size_t n_images = image_files.size();

  std::shared_ptr<TensorTransform> decode = std::make_shared<Decode>();
  std::shared_ptr<TensorTransform> hwc2chw = std::make_shared<HWC2CHW>();
  std::shared_ptr<TensorTransform> resize = std::make_shared<Resize>(std::vector<int>{256});
  std::shared_ptr<TensorTransform> centercrop = std::make_shared<CenterCrop>(std::vector<int>{224});
  std::shared_ptr<TensorTransform> normalize = std::make_shared<Normalize>(
    std::vector<float>{123.675, 116.28, 103.53}, std::vector<float>{58.395, 57.12, 57.375}
  );
  mindspore::dataset::Execute transform(
    std::vector<std::shared_ptr<TensorTransform>>{ decode, resize, centercrop, normalize, hwc2chw }
  );

  Status ret;
  std::vector<MSTensor> model_inputs = model.GetInputs();
  double total_time = 0;
  for (size_t i = 0; i < n_images; ++i) {
    struct timeval start = {0};
    struct timeval end = {0};
    std::vector<MSTensor> inputs;
    std::vector<MSTensor> outputs;
    std::cout << "\r[INFO] [" << 100.0*(i+1)/n_images << "%] Predicting " << image_files[i];

    MSTensor image = ReadFileToTensor(image_files[i]);
    transform(image, &image);
    inputs.emplace_back(model_inputs[0].Name(), model_inputs[0].DataType(), model_inputs[0].Shape(),
                        image.Data().get(), image.DataSize());
    gettimeofday(&start, nullptr);
    ret = model.Predict(inputs, &outputs);
    gettimeofday(&end, nullptr);
    if (ret != kSuccess) {
        std::cout << "[ERROR] Failed to predict " << image_files[i] << ", error: " << ret << std::endl;
        return 1;
    }
    total_time += 1e3 * (end.tv_sec - start.tv_sec) + 1e-3 * (end.tv_usec - start.tv_usec);
    WriteResult(image_files[i], outputs, FLAGS_output_path);
  }
  double average_time = total_time / n_images;
  std::cout << "\n[INFO] Inference completed!" << std::endl;
  std::cout << "[INFO] The number of images: " << n_images << std::endl;
  std::cout << "[INFO] Average time per image: " << average_time << "(ms)" << std::endl;
  return 0;
}
