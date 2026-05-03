# Point Cloud Part Segmentation: Comparing PointNetand DGCNN on Airplane Data
Comparing PointNet and DGCNN for 3D point cloud part segmentation 
on the ShapeNet airplane dataset.

## Setup
Create a virtual environment and install dependencies:
```bash
pip install -r requirements.txt
```
Download /data folder from [Google Drive link](https://drive.google.com/drive/folders/1yTi5VA7B8YhvnMYY3wrIkntJulDjxyHm?usp=drive_link), and place it in the root of this repository so the folder structure 
looks like:

```
shapenet/
  data/
    ShapeNet/
      processed/
        air_train.pt
        air_val.pt
        air_test.pt
        air_trainval.pt
  model.py
  model_var.py
  ...
```
## File Structure
- `model.py` — PointNet implementation
- `model_var.py` — DGCNN implementation
- `dataset.py` — data loading
- `train.py` — train PointNet
- `train_var.py` — train DGCNN
- `reproduce_results.py` — reproduce mIoU scores and visualizations

## Reproducing Results
Make sure the data is downloaded and the following files are in the root folder:
-  `checkpoint_best_base.pth`
-  `checkpoint_best_var.pth`
then run:

```bash
python reproduce_results.py
```

This will print the mIoU score for both models and save visualization 
figures to the root folder.
