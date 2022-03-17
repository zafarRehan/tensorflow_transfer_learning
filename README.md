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
<h3>Output</h3>
<img src="/images/out1.png" width=600/>

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
#### 1. In the notebook this code block was changed

<b>From</b>
```python
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
 ```

<b>To</b>
```python
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
```

Basically ```   int(member[5][1].text)  ``` is changed to ```   int(float(member[1][0].text))   ``` </br>
The reason is: </br></br>

<ul>
<li>In licence_detection annotation file the [bndbox] element was present at <b><i>sixth</i></b> position inside [object] element, whereas in the other annotation file it is present in <b><i>second</i></b> position.</li>
    
<li>In licence_detection annotation file the contents of [bndbox] element were of <b><i>int</i></b> type whereas in the other one it is of <b><i>float</i></b> type.</li>
</ul>
</br></br>
    
#### 2. In create_tfrecords.py 

<ul><li>Added dict at line: 32 </br>
    
    index_to_label = {1: 'car', 2:'pool'}
because unlike in the licence_detect annotation file we dont have class name as text in here so we need to change it to text from int</li>

<li>Changed line: 66</br> 
from 
    
    classes_text.append(row['class'].encode('utf8'))
to

    classes_text.append(index_to_label[row['class']].encode('utf8'))

for the same reason</li>
</ul>
</br>
</br>

### Directory Changes
Some source directory needs to be changed depending upon the folder structure of training data images which can be easily figured out when some error will be shown in Colab
</br></br>

### Running the Code
After making all these changes we are good to go and can proceed to run the code without issues. </br>

I made the exact same changes and prepared another Colab notebook for the above dataset here: <a href="/satellite_car_pool.ipynb">satellite_car_pool.ipynb</a>

### Result of training
<p><img src="/images/out2.png" width=300/>        <img src="/images/out3.png" width=300/></p>


The detection is not that good but also remember that this is the result of just an hour of training and also, you can see cars are getting detected pretty well but pools aren't. The reason being that the number of images with Pool is far less than images with Cars

    train['class'].value_counts()
    
    Output:
    1    11069
    2     2677
    Name: class, dtype: int64
Here 1 = Car, 2 = Pool
As can be seen above there are 11069 marked Cars in the training dataset whereas only 2677 Pools and this is called as <a href="https://www.analyticsvidhya.com/blog/2021/06/5-techniques-to-handle-imbalanced-data-for-a-classification-problem/">Imbalanced Dataset</a>. Though it is not a severe case of imbalance here are an article on <a href="https://towardsdatascience.com/having-an-imbalanced-dataset-here-is-how-you-can-solve-it-1640568947eb">how you can fix it.</a> 
</br></br>

### Additional Changes / Tuning
Remember <a href="/pipeline.config">pipeline.config</a>? This is the file which decides a model's configuration. Every model downloaded from <a href="https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md">Model Zoo</a> will have this file which can be edited to re-train the model as required.

<ul>
    
    model {
    ssd {
        num_classes: 1
<li>num_classes : It is a setting of the number to classify. It is written relatively at the top of the config file.</li>
 
    train_config: {
        batch_size: 32
        num_steps: 5000
        optimizer {
            momentum_optimizer: {
                learning_rate: {
                    cosine_decay_learning_rate {
                        total_steps: 5000
                        warmup_steps: 1000
    }
<li>batch_size :  This value is often the value of 2 to the nth power as is customary in the field of machine learning. And, the larger this value is, the more load is applied during learning, and depending on the environment, the process may die and not learn. The more the value the more RAM it will consume. </li>
    
<li>num_steps : The number of steps to learn. More the value better the model will train and more is the time required for training</li>
    
<li>total_steps and warmup_steps: I am investigating because it is an item that was not in the config of other models, total_steps must be greater than or equal to warmup_steps. (If this condition is not met, an error will occur and learning will not start.)</li>
    
</br>If you want In-depth knowledge of each configuration in pipeline.config <a href="https://neptune.ai/blog/tensorflow-object-detection-api-best-practices-to-training-evaluation-deployment"> Here it is </a>
    
</br></br>
## Choosing your model
Choice of model to perform transfer learning upon is the key for best results. </br>
Our data here was an average of 400px X 400px in licence dataset wheraes it was 224px X 224px for satellite_car_pool dataset. The base model I chose here was trained on images resized to 320px X 320px so this was perfect for their training.</br>

Now suppose you want to train a dataset of high res images say 1920px X 1080px. Training them on a model trained with 320X320 wont give excellent results.
</br>When you go to the Model Zoo every model has their size written with their name. Choose the nearest one to your dataset average size.


<h3 align=center>Thats all folks, go ahead and train your first Model!!</h3>

    
    
    
    
    
