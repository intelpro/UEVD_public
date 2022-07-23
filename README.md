# UEVD(ECCV 2022, Oral presentation)
**This repository is for the ECCV 2022 paper, "Event-guided Deblurring of Unknown Exposure Time Videos".**

\[[ArXiv](https://arxiv.org/pdf/2112.06988.pdf)\]
\[[ECCV2022]()\] 
\[[Supp]()\] 
\[[Oral(YouTube)]()\]
\[[Project](https://intelpro.github.io/UEVD/)\]

### Demo videos on real world blurry videos
<img src="https://github.com/intelpro/UEVD_public/raw/main/figure/video_results_real_blur.gif" width="60%" height="60%">
<!--
![real_blur_045_resized](/figure/video_results_real_blur.gif "real_blur_045_resized")
-->

### Demo videos on real-world event datset
<img src="https://github.com/intelpro/UEVD_public/raw/main/figure/video_results_real_event3.gif" width="60%" height="60%">
<!--
![real_event_045_resized](/figure/video_results_real_event3.gif "real_event_045_resized")
-->

## Color-DVS
#### The first public event-based deblurring dataset. Our dataset contain diverse scene including real-world event using color-DAVIS camera.
You can download the raw-data(collected frame and events) from this google drive [link](https://drive.google.com/file/d/16qlLDOm5Q6fqpDYxNYMtm7reGfzaOyra/view?usp=sharing)

Also, you can download the processed data for handling unknown exposure time videos [link](https://drive.google.com/file/d/1AxAwtZKP0NUCRUbiLpgZZ1ggrnLF2hbZ/view?usp=sharing)

## Installation
This code was tested with:
* pytorch 1.2.0
* CUDA 10.2
* Python 3.7 
* Ubuntu 18.04 using TITAN RTX GPU

```
pip install -r requirements.txt
bash install.sh
```

## Test
<!-- TBD: setup detail-->
```
python test_deblur_dvs.py --dataset 'dvs'
```


## Training
<!-- TBD: setup detail-->
```
python train_deblur_dvs.py --dataset 'dvs' --epochs 21 --batch_size 2 \
--test_batch_size 1 --use_multigpu True
```


## Results
* The quantitative comparisons are attached as belows for a reference.
<img src="https://raw.githubusercontent.com/intelpro/UEVD_public/main/figure/table2.png" width="70%" height="70%">

* The visual results of temporal activation map of the ETES modules on the vaious datasets.
<img src="https://raw.githubusercontent.com/intelpro/UEVD_public/main/figure/Figure5.png" width="70%" height="70%">


## Reference
> Taewoo Kim, Jeongmin Lee, Lin Wang, and Kuk-Jin Yoon" Event-guided Deblurring of Unknown Exposure Time Videos", In _ECCV_, 2022.

**BibTeX**
<!-- TBD: Change to ECCV bibtex format-->
```bibtex
@article{kim2021event,
  title={Event-guided Deblurring of Unknown Exposure Time Videos},
  author={Kim, Taewoo and Lee, Jungmin and Wang, Lin and Yoon, Kuk-Jin},
  journal={arXiv preprint arXiv:2112.06988},
  year={2021}
}
```

## Contact
If you have any question, please send an email me(intelpro@kaist.ac.kr)

## License
The project codes and datasets can be used for research and education only. 

