# GelSight_Wedge

 This repository is used for calibrating the GelSight Wedge Sensor and predict the contact pose of a Deformable Linear Object(DLO).
 
Firstly, change the ip address in config file. 
 
### Sensor Calibration
#### 1. Projective transformation
Extract the quacorners in one tactile image. Left mouse is used for selecting every corners, right mouse is used for remove last selection. 
```bash
python3 calibration/corner_select.py
```
Then replace the selected_corners in the config file. 
```bash
 selected_corners: [(36, 45), (46, 432), (584, 421), (566, 36)] # TL, TR, BR, BL
```

#### 2. Generate the look-up table
First use images_saver script to save a set of images. Press "s" for saving the images that will be saved in saved_imgs directory. 
```bash
python3 calibration/images_saver.py
```


Then use calibration script to generate the look-up table. 
```bash
python3 calibration/calibration.py
```
```bash
 -use "w, s, a, d" to adjust the circle position, use "m,n" to adjust the radius of the circle
 -press "Esc" after the adjustment 
```

### Pose Estimation
Following the example in Gelsight.py script, a object need to be generated 
```bash
IP = stream["ip"]
Size = (cropped_h, cropped_w)
gelsight = GelSight(IP, selected_corners, output_sz=Size)
gelsight.start()
```
Then the pose can be obatined from gelsight.pc.pose


### Marker Tracking(TBD)
