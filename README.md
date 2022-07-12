# UDUTV(ECCV 2022, Oral presentation)
**This repository is for the ECCV 2022 paper, "Event-guided Deblurring of Unknown Exposure Time Videos".**

\[[ArXiv](https://arxiv.org/pdf/2112.06988.pdf)\]
\[[ECCV2022]()\] 
\[[Supp]()\] 
\[[Oral(YouTube)]()\]
\[[Project](https://intelpro.github.io/UEVD/)\]

### Demo videos on real world blurry videos
![real_blur_045_resized](/figure/video_results_real_blur.gif "real_blur_045_resized")


## Color-DVS dataset for event-guided motion deblurring
#### The first public event-based deblurring dataset. Our dataset contain diverse scene including real-world event using color-DAVIS camera.
You can download the raw-data(collected frame and events) from this google drive [link](https://drive.google.com/file/d/16qlLDOm5Q6fqpDYxNYMtm7reGfzaOyra/view?usp=sharing)

Also, you can download the processed data for handling unknown exposure time videos [link](https://drive.google.com/file/d/1AxAwtZKP0NUCRUbiLpgZZ1ggrnLF2hbZ/view?usp=sharing)
## Installation
This code was tested with pytorch 1.2.0, CUDA 10.2, Python 3.7 and Ubuntu 18.04 using TITAN RTX GPU
```
pip install -r requirements.txt
bash install.sh
```
## Contact
If you have any question, please send an email me(intelpro@kaist.ac.kr)

## License
The project codes and datasets can be used for research and education only. 
