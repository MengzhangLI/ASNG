# MICCAI 2020: Automatic Data Augmentation for 3D Medical Image Segmentation

## Requirements
This framework is built based on framework of [nnUNet](https://github.com/MIC-DKFZ/nnUNet). Being familiar with its pipeline is **prerequisite** of ASNG.
- python 3
- torch >= 1.0
- Other packages such as tqdm, dicom2nifti, scipy, batchgenerators, numpy, sklearn, SimpleITK etc are listed in `./nnunet.egg-info`

 Due to version upgrade of nnUNet, it may encounter some troubles such as conflict between batchgenerators and numpy (https://github.com/MIC-DKFZ/nnUNet/issues/145). History version (before Febrary, 2019) of nnUNet is strongly recommended.

The original version nnUNet which ASNG utilized could be downloaded from [original_nnUNet.zip](https://pan.baidu.com/s/1Tm3wJgb7yH1di12I3rwpwg) with password: `ASNG`. Comparison between **original nnUNet** and **ASNG** may be helpful for debug.

## Usage
The same with nnUNet pipeline, dataset must be preprocessed as [MSD](http://medicaldecathlon.com/) default format.
Other information about MSD can be found from: https://decathlon-10.grand-challenge.org/Home/.
```
# create environment variables

export nnUNet_base='YOUR_PATH/ASNG/'
export nnUNet_preprocessed='YOUR_PATH/ASNG/nnUNet_preprocessed/'
export RESULTS_FOLDER='YOUR_PATH/ASNG/'

cd YOUR_PATH/ASNG/
rm -r nnUNet_raw_cropped/
rm -r nnUNet_raw_splitted/
cp -r nnUNet_raw/ nnUNet_raw_splitted/

# Task04 Hippocampus is small, very helpful for debug.

cd nnUNet_raw_splitted/Task04_Hippocampus/imagesTr/
for var in *.nii.gz; do mv "$var" "${var%.nii.gz}_0000.nii.gz"; done

cd YOUR_PATH/ASNG/nnunet/
python experiment_planning/plan_and_preprocess_task.py -t Task04_Hippocampus
python run/run_training.py 3d_fullres nnUNetTrainer Task04_Hippocampus 1 --ndet

```

## Notification
- Although ASNG is theoretically much more effectively than other reinforcement learning based algorithms in AutoML, it is still somewhat time-consuming in medical image segmentation tasks.
- Some records such as (1) training logs, (2) nnUNet plans making pickle files and (3) train/valid split infomation are stored at `./TrainingRecord/`
- The main difference between ASNG and nnUNet is from: `./nnunet/training/network_training/network_trainer.py`
where parameters update by Monte Carlo implements.

## Acknowledgments
The authors would like to thank Fabian Isensee for his great PyTorch implementation of [nnUNet](https://github.com/MIC-DKFZ/nnUNet).

## Citation
Including the following citation in your work would be highly appreciated:
```
Ju Xu, Mengzhang Li, and Zhanxing Zhu. "Automatic Data Augmentation for 3D Medical Image Segmentation." International Conference on Medical Image Computing and Computer-Assisted Intervention. Springer, Cham, 2020.
```