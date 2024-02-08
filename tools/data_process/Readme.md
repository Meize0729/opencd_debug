# Data process
Please go to this dir for data process
```shell script
cd tools/data_process
```
Next, go to the dataset you want to use and proceed with further processing. **I believe that these preprocessing steps and code will satisfy you and won't consume a lot of your time.**
```shell script
# must choose one
cd WHU_Building_Dataset / Inria_Building_Dataset / Levir_CD_Dataset / S2Looking_Dataset
```
**Notice:** 
1. You should replace **data_dir** with your specific path or specific the exact location of **data_dir** in the terminal beforehand.
2. All labels are saved in uint8 PNG format, where pixel value 255 represents buildings and pixel value 0 represents non-buildings.
3. Each dataset will eventually form a txt file called ***data_list***, and the data format saved in it will be as follows:
```
# building extraction
image, **, **, label, ** 
# change detection
image_a, image_b, label_cd, **, **
# both
image_a, image_b, label_cd, label_a, label_b
```