# Introduction

This repo is the source code of [Acemap-Paper X-Ray](<https://www.acemap.info/paper-x-ray>), which is a system to evaluate the look of your uploaded paper for a specific conference. 

For example, recently we release a AAAI2020 module to evaluate the authors' submissions of AAAI2020 and we receive about 4300 submissions. Before the AAAI2020 module, we have released INFOCOM2020 module as well.





# How it works

We use two models to evaluate the look of one paper - ResNet and Lightgbm, and the final score is a combination of these two scores.

## ResNet

We convert the pdf file to an image, like:

![](https://s2.ax1x.com/2019/09/09/ntW3UH.jpg)

then use ResNet to learn the features of the received papers.

The work in our experiments is similar to the work in (Deep Paper Gestalt) <sup>[1]</sup>  



## Lightgbm

We extract the information in the pdf file, like numbers of figures/tables/formulas, and then use lightgbm  to do the classification.





# Run

## preparation

### data

You need to prepare the pdf files for some conference, and put these files in the ```input/raw/``` folder.

```input/raw/conference```: The received papers of that conference in recent years

```input/raw/arxiv```: The rejected papers of that conference in recent years, like workshop papers. To get enough negative samples and to differentiate it from papers of other conferences, you can also put the 

### environment

The environment in our experiments is:

- system: Ubuntu 18.04.2

- python version: python 3.5.2

- required packages:

  ```
  PyPDF2
  fitz
  torch
  torchvision
  PIL
  sklearn
  numpy
  tqdm
  lightgbm
  pandas
  argparse
  ```

  Most of the packages are in Anaconda.

  

## train

Switch to ```scripts``` folder, and train by the command:

```
./train.sh
```

there are four files needed in the training procession:

```
lgb_process.py # extract the information from pdf files, like numbers of figures/tables/formulas
lgb_train.py # train the lightgbm model using the data extracted by lgb_process.py
nn_process.py # convert the pdf files to images
nn_train.py # train the nn model using images in the nn_train.py.
```

Then we will get two models:

```
lgb_model: output/lgb_output/lgb_model.txt
nn_model: output/nn_output/model_best.pth.tar
```



## predict

to predict a pdf file:

```
python predict.py --pdf_path [your_pdf_path]
```

and you will get a score for this conference.





**Reference: **

**[1] Huang J B. Deep Paper Gestalt[J]. arXiv preprint arXiv:1812.08775, 2018.** 