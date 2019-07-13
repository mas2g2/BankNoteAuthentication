# BankNoteAuthentication
This machine learning project uses logistic regression to train a model
to determine if a banknote is authentic.</br>

To determine the authenticity of a banknote we will be looking at the following 
features of each sample to predict their authenticity:</br>
1. variance of Wavelet Transformed image (continuous)</br>
2. skewness of Wavelet Transformed image (continuous)</br>
3. curtosis of Wavelet Transformed image (continuous)</br>
4. entropy of image (continuous) 

## Dataset info

Data were extracted from images that were taken from genuine and forged banknote-like specimens. For digitization, an industrial camera usually used for print 
inspection was used. The final images have 400x 400 pixels. Due to the object lens and distance to the investigated object gray-scale pictures with a resolution of 
about 660 dpi were gained. Wavelet Transform tool were used to extract features from images.
