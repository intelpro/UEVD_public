# UDUTV(ECCV 2022, Oral presentation)
**This is official repository of EDUTV(Event-guied Deblurring of Unknown Exposure Time Videos)**
\[[ArXiv](https://arxiv.org/pdf/2112.06988.pdf)\]
\[[ECCV2022]()\] 
\[[Supp.](https://openaccess.thecvf.com/content/ICCV2021/supplemental/Sim_XVFI_eXtreme_Video_ICCV_2021_supplemental.pdf)\] \[[Demo(YouTube)](https://www.youtube.com/watch?v=5qAiffYFJh8)\] \[[Oral12mins(YouTube)](https://www.youtube.com/watch?v=igwy1TJQiRc&t=13s)\] \[[Flowframes(GUI)](https://nmkd.itch.io/flowframes)\]

### Demo videos on real world blurry videos
![real_blur_045_resized](/figure/video_results_real_blur.gif "real_blur_045_resized")


## Color-DVS dataset for event-guided motion deblurring
#### The first public event-based deblurring dataset. Our dataset contain diverse scene including real-world event using color-DAVIS camera.
You can download the raw-data(collected frame and events) from this google drive [link](https://drive.google.com/file/d/16qlLDOm5Q6fqpDYxNYMtm7reGfzaOyra/view?usp=sharing)

Also, you can download the processed data for handling unknown exposure time videos [link](https://drive.google.com/file/d/1AxAwtZKP0NUCRUbiLpgZZ1ggrnLF2hbZ/view?usp=sharing)
## Requirements
Our code is tested using PyTorch1.2 and tested under following condition:
* Python 3.7
* Pytorch 1.2.0
* Cuda 10.2
* NVIIDIA TITAN RTX 
* Ubuntu 18.04