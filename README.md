# Allweather-Denoising
![](assets/test_results.jpeg)

## Introduction
Adverse weather conditions can significantly degrade object detection performance in autonomous driving systems, leading to unreliable model decisions. This project addresses that problem by developing a generative AI-based image restoration filter designed to remove weather-related degradations from vehicle camera images.


## Dataset Construction
To keep the scope aligned with autonomous driving applications and improve model relevance, the study focuses exclusively on in-vehicle camera data. For snowy conditions, [VideoDesnowing](https://github.com/haoyuc/VideoDesnowing) dataset was used as the foundation. From this dataset, 22 distinct driving scenarios were separated, each containing different environmental settings, camera perspectives, and in-car visual compositions. Based on the clean reference images, additional synthetic adverse-weather datasets were generated: rainy images were created using [SyRaGAN](https://github.com/jaewoong1/SyRa-Synthesized_Rain_dataset) and [VRGNet](https://github.com/hongwang01/VRGNet) under varying angles and intensities, while foggy images were produced with [FINet](https://github.com/zhangzhengde0225/FINet) using different depth and density settings.


## Dataset Structure

| Weather Condition | Training Sample | Training Scenes | Test Sample | Test Scenes|
|---|---:|---:|---:|---:|
| Snow | 1956 | 19 | 600 | 3 |
| Rain | 1956 | 19 | 600 | 3 |
| Fog | 1956 | 19 | 600 | 3 |



The resulting synthetic dataset, covering snow, rain, and fog, was used to train a patch-based diffusion model for adverse weather removal. The training split for each weather condition contains 1,956 images from 19 scenarios, while the test split contains 600 images from 3 unseen scenarios. This scenario-based separation enables evaluation under previously unseen environmental conditions, camera viewpoints, and vehicle-interior contexts, providing a more realistic measure of model generalization.

