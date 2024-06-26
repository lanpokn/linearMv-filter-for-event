# Event Camera Demo
**By Cedric Scheerlinck**
cutoff fre to 20, img is just 0.5, then final img looks good in pure event output. I can use it to estimate event data. first I need to use it to generate my intensity change map! initial L0 is background

you can either change L back to Y or not. better not, thus you can minus it and divide with 0.1(same with 0.1*p)

Load and visualize events in a jupyter notebook.  
Inspired by [https://github.com/cedric-scheerlinck/dvs_image_reconstruction](https://github.com/cedric-scheerlinck/dvs_image_reconstruction).

![filter_pic](images/teaser.png)

## Installation
Install required packages:
```pip install -r requirements.txt```

Enable ipywidgets:
```jupyter nbextension enable --py widgetsnbextension```

## Run
```jupyter notebook main.ipynb```

## Video
[![dvs_image_reconstruction_video](images/thumbnail_combined.png)](https://youtu.be/bZ0ZKido0Ag)
[https://youtu.be/bZ0ZKido0Ag](https://youtu.be/bZ0ZKido0Ag)

### Reference
* Cedric Scheerlinck, Nick Barnes, Robert Mahony, "Continuous-time Intensity Estimation Using Event Cameras", Asian Conference on Computer Vision (ACCV), Perth, 2018.  
[PDF](https://cedric-scheerlinck.github.io/files/2018_scheerlinck_continuous-time_intensity_estimation.pdf), [Website](https://cedric-scheerlinck.github.io/continuous-time-intensity-estimation), [BibTex](https://cedric-scheerlinck.github.io/files/2018_accv_continuous_bibtex.txt).
