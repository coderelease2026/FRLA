# Forgetting-Resistant and Lesion-Aware Source-Free Domain Adaptive Fundus Image Analysis with Vision-Language Model
This is an anonymous repository for our method FRLA based on PyTorch, containing the environment installment, training instructions, source domain model weight, and final target domain model weight. 

## Environment
To use the repository, we provide a conda environment.
```bash
conda env create -f environment.yml
conda activate frla
```

## Datasets
[ODIR](https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k) (Not required if using our pretrained source model weight)

[FIVES](https://figshare.com/articles/figure/FIVES_A_Fundus_Image_Dataset_for_AI-based_Vessel_Segmentation/19688169/1)

- Download the above datasets. For the compatibility with [FLAIR](https://github.com/jusiro/FLAIR)'s naming convention, name the folder of FIVES as `13_FIVES`. The prepared data directory would look like:
```bash
├── Your data directory
    ├── 13_FIVES
        ├── train
        ├── test
```
- Modify the `PATH_DATASETS` in `./FLAIR/local_data/constants.py` to your folder containing the `13_FIVES` folder.
- Modify the path prefixes of images in the '.txt' files under the folder './data/fundus_4c/'. In addition, the class name file is also under the folder './data/fundus_4c/'.


## Training
The following are the steps for the ODIR to FIVES adaptation.
### Source domain
- Download our pre-trained source model weights from [here](https://drive.google.com/drive/folders/1Tel5Lyiy6EXSHCMXi3d8mJpb9ABVW-kS?usp=sharing) (recommended for the reproduction of our results). Or, you can also run the following command to train the source domain model:
```bash
CUDA_VISIBLE_DEVICES=0 python image_target_of_oh_vs.py --cfg "cfgs/fundus_4c/source.yaml" SETTING.S 0
```

- Put the source model weights under `./Modelzoom/source/uda/fundus_4c/O/`.

### Target domain
- Download the FLAIR model weight from [here](https://drive.google.com/file/d/1bA3PUTfXj-DorfNZ6EHrr1cYD0cMvVJ-/view?usp=sharing), and put it under `./FLAIR/flair/modeling/flair_pretrained_weights/`.
- Run the following command to execute source-free domain adaptation.
```bash
CUDA_VISIBLE_DEVICES=0 python image_target_of_oh_vs.py --cfg "cfgs/fundus_4c/patch.yaml" SETTING.S 0 SETTING.T 1
```
## Result
Accuracy = 80.38%

97.0 95.5 61.0 68.0

## Model weights
For transparency, we provide the final target domain model weights after training of our method and the compared methods.
+ SHOT [weight](https://drive.google.com/drive/folders/1kaP7NpdvJMHGjvlpfkuXNIGOouYPd25t?usp=sharing)
+ COWA [weight](https://drive.google.com/drive/folders/1A79-Q2onDK4pyvTBXiOWnu-_chPDvuZs?usp=sharing)
+ DIFO [weight](https://drive.google.com/drive/folders/1Sb8Q5h8ERbxRERHfTgwtMLnciLmyKbh3?usp=sharing)
+ FRLA [weight](https://drive.google.com/drive/folders/1fkvXx8ppc3XvouzE7xEhv2NYnpK__q4T?usp=sharing)
