# Breast Cancer Classification Using Machine Learning
**By: Eddie Olmos**

## Table of Contents

[Executive Summary](#executive-summary) <br>
[Problem Statement](#problem-statement) <br>
[Background Information](#background-information) <br>
[Dataset Used](#dataset-used) <br>
[Data Dictionary](#data-dictionary) <br>
[Software Requirements](#software-requirements) <br>
    [Project Workflow](#project-workflow) <br>
[The Data Science Process](#the-data-science-process) <br>
[Data Cleaning, EDA, and Preprocessing](#data-cleaning-eda-and-preprocessing) <br>
[Modeling](#modeling) <br>
[Evaluation and Conceptual Understanding](#evaluation-and-conceptual-understanding) <br>


### Executive Summary
Upon implementing the revisions, my model was still performing better on the training data than the testing data, but it took about 20 times as many epochs for the training accuracy to get close to 100% accuracy.  For the first 50 epochs the validation data was actually performing better than the training data signifying an impactful reduction in how overfit the model was.  After 200 epochs, the model’s validation accuracy stood at 82% and even reached about 85% around 125 epochs.  The train and validation loss was also significantly lower than the pervious version of the model which was also a good indication of the models improvement.

**Next Steps**

There will always be welcomed room for improvements on this initiative and I look forward to continuing to build upon my work moving forward.  This includes but is not limited to:

> Implementing a GridSearch to tune hyper parameters.<br>
> Building a confusion matrix to see which classes are performing better than others.<br>
> Connect with an oncologist or diagnostic medical sonographer to answer a multitude of  questions that arose throughout my project.<br>

### Problem Statement
Build and train a Convolutional Neural Network (CNN) to classify a breast ultrasound as either normal, benign, or malignant.

For my capstone project, I wanted to see if I could build and train a machine learning algorithm called a Convolutional Neural Network (or CNN), that could classify a breast ultrasound as either normal, benign, or malignant.  Now I’ll be the first one to acknowledge the sensitive nature and gravity of this topic along with my lack of expertise in the medical field, but I think we can all agree it’s much more rewarding to be part of an initiative we’re truly passionate about.  And considering the age of information we live in, there are vast datasets offered by the real medical professionals available to the pubic to study and explore these correlations.

### Background Information
About 1 in 8 U.S. women (or 13%) will develop invasive breast cancer over the course of her lifetime.  I’m sure each and and everyone of us has had a close family member or friend that has embarked on this journey.  Personally, one of my closest cousins required a double mastectomy on her path to recovery and one of my best friend’s mother who already survived beast cancer once in 2009 detected another growth earlier this year.  Thankfully, they are now both healthy and well, but the reality is about 43,600 women in the U.S. are expected to die in 2021 alone from breast cancer. Death rates have been steady in women under 50 since 2007, but have continued to drop in women over 50. The hopeful news is the overall death rate from breast cancer decreased by 1% each year from 2013 to 2018. These decreases are thought to be the result of treatment advances and earlier detection through screening. <sup>1</sup>

Breast cancer is a type of cancer that starts in the breast. Cancer starts when cells begin to grow out of control.  Breast cancer cells usually form a tumor that can often be seen on an x-ray or felt as a lump. Breast cancer occurs almost entirely in women, but men can get breast cancer, too.  It’s important to understand that most breast lumps are benign and not cancer (malignant). Non-cancerous breast tumors are abnormal growths, but they do not spread outside of the breast. They are not life threatening, but some types of benign breast lumps can increase a woman's risk of getting breast cancer. Any breast lump or change needs to be checked by a health care professional to determine if it is benign or malignant (cancer) and if it might affect your future cancer risk. <sup>2</sup>

The series of tests needed to evaluate a possible breast cancer usually begins when a woman or their doctor discover a mass or abnormal calcifications on a screening mammogram, or a lump or nodule in the breast during a clinical or self-examination. Less commonly, a woman might notice a red or swollen breast or a mass or nodule under the arm.  The following tests may be used to diagnose breast cancer or used for follow-up testing after a breast cancer diagnosis. <sup>3</sup>

The technique used to do these X-rays is called mammography. The images may show a small tumor that cannot be felt during an examination or other breast changes. Diagnostic mammography is similar to screening mammography except that more pictures of the breast are taken. It is often used when a woman is experiencing signs, such as a new lump or other symptoms. Diagnostic mammography may also be used if something suspicious is found on a screening mammogram.  Up to 10% to 15% of the time, mammography will not show an existing cancer, called a "false-negative" result.  <sup>3</sup>

An MRI uses magnetic fields, not x-rays, to produce detailed images of the body. A special dye called a contrast medium is given before the scan to help create a clear picture of the possible cancer. This dye is injected into the patient’s vein. Breast MRI is also a screening option, along with mammography, for some women with a very high risk of developing breast cancer and for some women who have a history of breast cancer. <sup>3</sup>

An ultrasound uses sound waves to create a picture of the breast tissue. An ultrasound can distinguish between a solid mass, which may be cancer, and a fluid-filled cyst, which is usually not cancer. <sup>3</sup>

For most types of cancer, a biopsy is the only sure way for the doctor to know if an area of the body has cancer.  A biopsy is the removal of a small amount of tissue for examination under a microscope. Other tests can suggest that cancer is present, but only a biopsy can make a definite diagnosis. A pathologist then analyzes the sample(s). A pathologist is a doctor who specializes in interpreting laboratory tests and evaluating cells, tissues, and organs to diagnose disease. <sup>3</sup>

Examination of the tumor under the microscope is used to determine if it is invasive or non-invasive (in situ); ductal, lobular, or another type of breast cancer; and whether the cancer has spread to the lymph nodes. <sup>3</sup>

### Dataset Used
The Breast Ultrasound Images Dataset (or BUSI Dataset) was collected by Dr. Aly Fahmy and his team who is a Professor of Artificial Intelligence and Machine Learning. He was the Ex-Dean of the Faculty of Computers and Information at Cairo University. Currently, he is the Dean of the Artificial Intelligence College affiliated to the Arab Academy for Science and Technology.

The data collected at baseline include breast ultrasound images among women in ages between 25 and 75 years old. This data was collected in 2018 and was taken from 600 female patients. The dataset consists of 780 images with an average image size of 500*500 pixels. The images are in PNG format. The ground truth images are presented with original images. The images are categorized into three classes, which are normal, benign, and malignant. <sup>4</sup>

### Data Dictionary

| normal | benign | malignant |
| --- | --- | --- |
| PNG | PNG | PNG |
| Breast ultrasounds classified as normal, or without tumors. | Breast ultrasounds classified as benign, or tumors that are not cancerous. | Breast ultrasounds classified as malignant, or tumors that **are** cancerous. |

### Software Requirements
> Google Colab <br>
> matplotlib.pyplot <br>
> numpy <br>
> PIL <br>
> tensorflow <br>
>> keras <br>
>>> layers <br>
>>> Sequential <br>
>>> image_dataset_from_directory <br>
> PIL <br>
> os <br>

### Project Workflow
To reproduce my .ipynb you will need to either [download](https://scholar.cu.edu.eg/Dataset_BUSI.zip) the image dataset and manually create a "datasets" directory along with a "test_data" directory, or you can download the folders from my drive using the links below:

Datasets: https://drive.google.com/drive/folders/1c8O6PS9kjjVNQ75DbBixHhJJDWzGKRWW?usp=sharing
Test Data: https://drive.google.com/drive/folders/1JMKN5haYZ_QKhyMK2iJWV9zEeTD2snIK?usp=sharing

A few lines of code may need to be updated to the location where you save these directories.  Once all paths are updated, the code should be able to run without any errors.

### The Data Science Process

#### Data Cleaning, EDA, and Preprocessing
There wasn’t very much data cleaning to do since the dataset was compiled by machine learning authorities.  I imagine there was plenty of heavy lifting required before the data got to me and for that I tip my hat to Dr. Fahmy and his team.  I manually calculated the average width and height of the 780 pictures and noticed my calculations deviated a bit from the description provided by the assemblers.  They stated the average width and height was 500x500 pixels but my calculations yielded approximately 500x615 pixels.  I also noticed they provided ground truth image masks in black and white.  My last significant observation was some of the malignant and benign ultrasounds contained anchor points and text images overlaid on the ultrasounds which will most likely affect results.  Although the masks didn’t contain these extra features, I assumed some sort of additional manipulation was required to generate the ground truth masks.  Also, the masks of the normal images were solid black which would make them relatively easy to classify.  Therefore, as an additional challenge I decided to remove the masks from the dataset directories and stick as close as I could the the raw ultrasounds.

For my CNN model, I utilized the Keras package from TensorFlow since it already had a really convenient method able to create test and validation datasets to train my model on from the sub directories they were already placed in by Dr. Fahmy’s team.  One of the requirements in order for a CNN model to process and train from an image dataset is that the images must be transformed into a uniform height and width.  I plotted a histogram to see each image’s width and height, and also ran some preliminary tests to see how reshaping the images was affecting the results.  There seems to be a lot of debate on how to handle this common deviation among images, some folks think you should resize to the dimensions of the smallest image, others say to resize to the largest. I received the best results by using my calculated average height and width, so that’s how I decided to proceed.  At this point, some of you may have noticed the breast cancer ribbon shown on this slide is a bit pixelated and the image isn’t very sharp.

#### Modeling
To create the model, I initialized and built the CNN layer by layer.  It required some manipulation of the pixels to be match the format expected by the CNN, and adding the convolutional layers to identify relationships among the features of the images.  It also required a few dense layers to process the output of the convolutional layers and finally transform the resulting throughput into one of our three classes.  Once our initial model was constructed, the time finally came to train the model with the subset of images set aside in the preprocessing stage and test it on the validation dataset.

Manually fine tuning the model to produce decent results was an incredibly tedious process that required copious research into how the convolutional layers worked and studying the affects the hyper-parameters were having on performance.  Nonetheless, I ended up with this first version of the model.

Within the first 10 times we ran our training data through our model, also referred to as our first 10 epochs, training accuracy was almost at 100% while the validation accuracy was only at about 75%.  This indicated our model was severely overfit, meaning it was doing great on the images it was being trained on, but no so well on unseen data.  Given the incredibly severe impact a false positive, or even worse a false negative would have on patients, I had to do better.  The reality is even trained medical professionals with decades of experience usually can’t definitively tell if a tumor detected on an ultrasound is a benign or malignant tumor until performing a biopsy.  Regardless, machine learning is an iterative process so it was time to go back to the drawing board.

To reduce the overfitting, I implemented some data augmentation that flips and rotates the images in ways that keep the model on its toes and attempts to change the patterns being identified by the convolutional layers.  I also added some dropout layers which drops certain nodes in the model and is another best practice to balance overfitting.  

#### Evaluation and Conceptual Understanding
Upon implementing the revisions, our model was still performing better on the training data than the testing data, but it took about 20 times as many epochs for the training accuracy to get close to 100% accuracy.  As you can see, for the first 50 epochs the validation data was actually performing better than the training data signifying an impactful reduction in how overfit the model was.  After 200 epochs, the model’s validation accuracy stood at 82% and even reached about 85% around 125 epochs.  The train and validation loss was also significantly lower than the pervious version of the model which was also a good indication of the models improvement.

Regardless, there will always be welcomed room for improvements on this initiative and I look forward to continuing to build upon my work moving forward this includes but is not limited to:

> Implement a GridSearch to tune hyperparameters. <br>
> Build confusion matrices to see which classes are performing better than others. <br>
> Connect with an oncologist or diagnostic medical sonographer to answer a multitude of  questions that arose throughout my project. <br>

### Sources

#### References
1. https://www.breastcancer.org/symptoms/understand_bc/statistics <br>
2. https://www.cancer.org/cancer/breast-cancer/about/what-is-breast-cancer.html <br>
3. https://www.cancer.net/cancer-types/breast-cancer/diagnosis <br>
4. https://scholar.cu.edu.eg/?q=afahmy/pages/dataset <br>
-Dataset provided by: Al-Dhabyani W, Gomaa M, Khaled H, Fahmy A. Dataset of breast ultrasound images. Data in Brief. 2020 Feb;28:104863. DOI: 10.1016/j.dib.2019.104863. <br>

#### Images
-https://oakland.edu/medicine/news/auto-list-news/2021/More-needs-to-be-done-to-help-Black-women-fight-breast-cancer,-says-OUWB-prof <br>
-https://chicago.suntimes.com/well/2021/10/8/22704713/breast-cancer-subtypes-treatments-developments-wellness <br>
-https://www.uptodate.com/contents/image/print?imageKey=PI%2F53453 <br>
-https://www.nationalbreastcancer.org/breast-ultrasound <br>
-https://www.sanovadermatology.com/skin-cancer-blog-cat/what-is-a-skin-biopsy/ <br>
-https://www.istockphoto.com/vector/breast-cancer-awareness-with-realistic-pink-ribbon-on-a-white-background-women-gm1176663746-328162763 <br>
-https://www.mdpi.com/1424-8220/10/9/8363/htm <br>
-https://blog.hellojs.org/create-a-very-basic-loading-screen-using-only-javascript-css-3cf099c48b19 <br>
-https://rethinkbreastcancer.com/women-with-mbc-need-you-to-be-their-ally/ <br>
-https://www.nasa.gov/centers/wstf/about_us/safety_and_mission_assurance/continual_improvement.html <br>
