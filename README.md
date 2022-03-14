# Tensorflow Transfer Learning
This repository explains how to perform transfer learning on any tensorflow pre-trained object-detection model.</br>
Any model listed in <a href="https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md">Model Zoo</a> can be re-trained using this tutorial.

## Why Transfer Learning?
Training a model to solve real world object detection problems is no easy task. It needs a lot of computing resources and time to train such models from scratch. </br>
Using transfer learning we can use the existing weights of the pre-trained models and change just the last few layers to customize it to fit our own problem domain. </br>
These <a href="https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md">models</a> are probably trained on super-computers which is impossible for many low to medium scale organizations to access or to afford. </br>

I trained my licence detection model in less than 3 hours on Google Colab and used the output model to detect licence plates on images here: https://github.com/zafarRehan/licence_plate_detection

Now let's jump into using the code.

## Running the default code in Colab
The repository contains the Notebook <a href="/license_plate_detection.ipynb">license_plate_detection.ipynb</a> which can be downloaded and executed directly on Google Colab.
Everything is pre-feeded in the Notebook, from datset to configuration files. </br>
Just click on <b>Runtime -> Run all</b> then sit back and relax and watch your custom model being built.</br>

The dataset I used here is from Kaggle https://www.kaggle.com/andrewmvd/car-plate-detection which contains 432 annotated images of cars with licence plates.</br>

The code is well-commented so each step is explained in comments in the code.

