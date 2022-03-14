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

</br>
</br>

## Training your own Model
Our main goal here is to train our own Object Detection model with excellent performance and in no time.</br>

First and foremost we need data to train our model on. You can download any annotated dataset from Kaggle, or <a href="https://towardsai.net/p/computer-vision/50-object-detection-datasets-from-different-industry-domains">here</a> or anywhere on the Internet.</br>

You can create your own dataset for object detection for which you must have: </br>
1. Atleat 300 to 400 images containing the object(s)
2. Annotating tool to draw the bounding boxes of the object(s), for example: https://www.youtube.com/watch?v=Tlvy-eM8YO4 (Recommended)</br></br>

## Changes to be made for Custom Training
As the problem changes so does varoius other parameters.</br>

In order to demonstrate the changes I will take another example to walk you through the changes to be made and the challenges that can be faced while changin them.

<b>Dataset Used : </b> https://www.kaggle.com/kbhartiya83/swimming-pool-and-car-detection

This dataset consist of 2 classes: </br> 
1. Car <br>
2. Swimming Pool

unlike the licence_plate_detection which has only one class <b> Licence </b> </br>

To handle this change in number of classes following changes must me made in: </br>
### custom.pbtxt
<table>
<tr>
<td width=400>
Before:

    item
    {
        id :1
        name :'licence'
    }
    
    
    
    
</td>
<td width=400>
After:

    item
    {
        id :1
        name :'car'
    }
    item
    {
        id: 2
        name: 'pool'
    }
    
</td>
</tr>
</table>
Note: The number of item should match number of classes in your dataset with proper name. </br>
<h3>pipeline.config </h3>
at line 3:</br>
<table>
<tr>
<td width=400>
change

    num_classes: 1
</td>
<td width=400>
to

    num_classes: 2
</td>
</tr>
</table>
    
Note: The value of num_classes must be equal to number of classes / different objects to be detected in your dataset. (Here: 'Car', 'Pool').</br>

</br>

### Annotation Changes
The annotation files can be different for different datasets or your own created annotation.
Let's compare our annotation file for the 2 datasets: </br>

<table>
<tr>
<td width=400>
<b>1. Licence Plate Annotation File</b>

    <annotation>
        <folder>images</folder>
        <filename>Cars2.png</filename>
        <size>
            <width>400</width>
            <height>400</height>
            <depth>3</depth>
        </size>
        <segmented>0</segmented>
        <object>
            <name>licence</name>
            <pose>Unspecified</pose>
            <truncated>0</truncated>
            <occluded>0</occluded>
            <difficult>0</difficult>
            <bndbox>
                <xmin>229</xmin>
                <ymin>176</ymin>
                <xmax>270</xmax>
                <ymax>193</ymax>
            </bndbox>
        </object>
    </annotation>      
</td>   
<td width=400>
<b>2. Satellite Car Pool Annotation File</b></br>

    <?xml version="1.0"?>
    <annotation>
        <filename>000000001.jpg</filename>
        <source>
            <annotation>ArcGIS Pro 2.1</annotation>
        </source>
        <size>
            <width>224</width>
            <height>224</height>
            <depth>3</depth>
        </size>
        <object>
            <name>1</name>
            <bndbox>
                <xmin>58.47</xmin>
                <ymin>40.31</ymin>
                <xmax>69.58</xmax>
                <ymax>51.43</ymax>
            </bndbox>
        </object>
        <object>
            <name>1</name>
            <bndbox>
                <xmin>10.32</xmin>
                <ymin>93.68</ymin>
                <xmax>21.43</xmax>
                <ymax>104.80</ymax>
            </bndbox>
        </object>
    </annotation>
</td>
</tr>
</table>

Well there is difference right? </br>
To handle these changes following code were changed: </br>
1. In the notebook this code block was changed

<table>
<tr>
<td width=400>
<b>From</b>

    import os
    import glob
    import pandas as pd
    import xml.etree.ElementTree as ET

    def xml_to_csv(path):
        xml_list = []
        for xml_file in glob.glob(path + '/*.xml'):
            tree = ET.parse(xml_file)
            root = tree.getroot()
            for member in root.findall('object'):
                value = (root.find('filename').text,
                         int(root.find('size')[0].text),
                         int(root.find('size')[1].text),
                         member[0].text,
                         int(member[5][0].text),
                         int(member[5][1].text),
                         int(member[5][2].text),
                         int(member[5][3].text)
                         )
                xml_list.append(value)
        column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
        xml_df = pd.DataFrame(xml_list, columns=column_name)
        return xml_df     
</td>   
<td width=400>
<b>To</b>

    import os
    import glob
    import pandas as pd
    import xml.etree.ElementTree as ET

    def xml_to_csv(path):
        xml_list = []
        for xml_file in glob.glob(path + '/*.xml'):
            tree = ET.parse(xml_file)
            root = tree.getroot()
            for member in root.findall('object'):
                value = (root.find('filename').text,
                         int(root.find('size')[0].text),
                         int(root.find('size')[1].text),
                         member[0].text,
                         int(float(member[1][0].text)),
                         int(float(member[1][1].text)),
                         int(float(member[1][2].text)),
                         int(float(member[1][3].text))
                         )
                xml_list.append(value)
        column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
        xml_df = pd.DataFrame(xml_list, columns=column_name)
        return xml_df
</td>
</tr>
</table>

