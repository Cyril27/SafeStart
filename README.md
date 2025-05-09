# SafeStart

Improvements in image processing and artificial intelligence techniques in recent years have opened the door to alcohol screening using gait analysis. However, human shape segmentation from real-word recordings faces a number of obstacles, including diverse background contents, uneven illumination or even the presence of shadows, which prevents proper generalization of those models. Today, we propose SafeStart, an efficient pipeline for robust gait analysis is addressed using a combination of image processing tools. Proper method to estimate the background is first investigated, which is used to highlight the moving body. Study of the best performing pipeline to extract the binary silhouette is then studied, offering promising results that can serve as efficient pre-processing before feeding the transformed data into different AI models.

# Repository structure
The repository contains all the Python code needed to process the data used to create and validate SafeStart. A brief description of each python file is given below. \
**define_background** : Explore best background using RGB/HSV/grayscale + Blurring/Not blurring configurations\
**backgroung_blurring_cross** : Investigate background subtraction in grayscale for background blurred/not blurred and frame blurred/not blurred\
**plot_bg_metric** : Display AGE and pSNR for different background estimations\
**ref_mask** : Allows manual definition of ground truth binary masks on given frame\
**main_pipeline** : Apply the silhouette extraction on a given frame using Method 1/2/3 \
**plot_skeleton_metric** : Display Jaccard Index and Dice Score for Method 1/2/3\
**output_video** : Creates the output video withan overlay of the binary mask and the skeleton\
**plot_method_metric** :Display skeleton metrics to compare methods 1 and 2\
