# AffordPoseNet
### AffordPose: A Large-scale Dataset of Hand-Object Interactions with Affordance-driven Hand Pose

## Requirements
This package has the following requirements:

* [Pytorch>=1.1.0](https://pytorch.org/get-started/locally/) 
* Python >=3.6.0
* [pytroch3d >=0.2.0](https://pytorch3d.org/) 
* [MANO](https://github.com/otaheri/MANO) 
* [bps_torch](https://github.com/otaheri/bps_torch) 
* [psbody-mesh](https://github.com/MPI-IS/mesh) (for visualization)

## Installation

To install the dependencies please follow the GrabNet steps: https://github.com/otaheri/GrabNet

## Getting started

## Examples

After installing the *AffordPoseNet* package, dependencies, and downloading the data and the models from AffordPose website, you should be able to run the following examples:


- #### Generate several grasps for new unseen objects
    
    ```Shell
    python grabnet/tests/grab_new_objects.py --obj-path $NEW_OBJECT_PATH \
                                             --rhm-path $MANO_MODEL_FOLDER
                                            

- #### Generate grasps for test data and compare to ground truth (GT)
    
    ```Shell
    python grabnet/tests/test.py     --rhm-path $MANO_MODEL_FOLDER \
                                     --data-path $PATH_TO_GRABNET_DATA
    ```

- #### Train AffordPoseNet with new configurations 
    
    To retrain AffordPoseNet with a new configuration, please use the following code.
    
    ```Shell
    python train.py  --work-dir $SAVING_PATH \
                     --rhm-path $MANO_MODEL_FOLDER \
                     --data-path $PATH_TO_GRABNET_DATA
    ```




## Citation

```
@InProceedings{Jian_2023_ICCV,
    author    = {Jian, Juntao and Liu, Xiuping and Li, Manyi and Hu, Ruizhen and Liu, Jian},
    title     = {AffordPose: A Large-Scale Dataset of Hand-Object Interactions with Affordance-Driven Hand Pose},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {14713-14724}
}
```
```
@inproceedings{GRAB:2020,
  title = {{GRAB}: A Dataset of Whole-Body Human Grasping of Objects},
  author = {Taheri, Omid and Ghorbani, Nima and Black, Michael J. and Tzionas, Dimitrios},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year = {2020},
  url = {https://grab.is.tue.mpg.de}
}
```

