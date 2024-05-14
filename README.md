This repository for the compression of three Convolutional Neural Network models—Lenet, Alexnet, and VGG—for classifying marine animal datasets. The compressed models presented in this project not only maintain the same accuracy as the original models but also significantly reduce their memory foot-print. This work was implemented for **Deep learning** final project at NYU Tandon 2024.
# Authors:
- Raksi Kopo (rk4585@nyu.edu)
- Duc Anh Van (dv2223@nyu.edu)
- Remon Roshdy (rr3531@nyu.edu)
# Deep Compression
Compressing Convolutional Neural Network models with pruning, trained quantization
It provides an implementation of the two core methods:

- Pruning
- Quantization

These are the main results on the [Sea animal](https://www.kaggle.com/datasets/vencerlanz09/sea-animals-image-dataste) datasets

  - Our experiments revealed the effectiveness of memory and computationally efficient techniques for CNNs in marine animal classification.
  - Model pruning led to significant reductions in model size across LeNet, AlexNet, and VGG16. For example, after pruning, the size of the models decreased by approximately 20% for LeNet, 35% for AlexNet, and 40% for VGG16, while maintaining competitive accuracy.
  - Fine-tuning pruned models demonstrated their adaptability, with accuracy levels successfully restored post-pruning. After fine-tuning, LeNet achieved an accuracy of 21% following a 60% pruning, AlexNet improved to 79% accuracy from an initial 45\% after a similar pruning, and VGG16 maintained its accuracy at 81% following a 60% pruning.
  - Quantization proved promising for model compression, achieving significant reductions in size with minimal impact on accuracy.
  - The most optimized model, in terms of size reduction and retained accuracy, is the quantized AlexNet, which has been reduced to 57.1 MB (four times smaller than the original) and maintains relatively good accuracy (72%). 

## Requirements
  - pytorch
  - pytorch-lightning
  - torchmetrics
  - torchvision
  - ipykernel
  - jupyter
  - matplotlib
  - numpy
  - scipy
  - scikit-learn
  - tqdm
  - tensorboard

## Project Structure
  ```
  model-compression/
  │
  ├── notbook - main script to run
  ├── Data/ directory containing 14,000 sea animals images of 23 species 
  │   
## TODOs
- [ ] Run on Jupiter 

## References
[[1]](https://arxiv.org/pdf/1510.00149v5.pdf) Han, Song, Huizi Mao, and William J. Dally. "Deep compression: Compressing deep neural networks with pruning, trained quantization and huffman coding." arXiv preprint arXiv:1510.00149 (2015)


## Acknowledgments
- [Pytorch](https://pytorch.org/docs/stable/nn.html#module-torch.nn.utils) for pruning and quantization library
- [Datasets](https://www.kaggle.com/datasets/vencerlanz09/sea-animals-image-dataste) for sea animal data

