# Requeriment
  - python 3.6
  - matplotlib
  - tensorflow 2.0
  - scipy
  - sklearn
  - seaborn


# How to run 

We have two main files part1.py and part2.py both of them contain everything from parameters to the main functionalities. 

In terms of directory structure it expect to have access to the dataset in the same directory and have another directory call output where it will store the models in hdf5, the images and a json with the results.

## 	Params
In terms of parameters I only play a bit with the basic ones like batch size and learning rate. The naming is mostly self explanatory minus in part1 where there is a parameter call params which control the options for the diferent parts. It is done that way to help generate bucles to try diferent parameters (this is omitted in the final version).

  
part1:
The experiment's names goes as follow:
  - 1_1:Baseline model params:[False,False,False]
  - 1_2:Dropout params:[True,False,False]
  - 1_3:Data augmentation params: [False,True,False]
  - 1_4:Batch normalisation  params: [False,False,True]
  - 1_5:Best model params: [True,True,True]
```python
experiment='1_5_234' #name of the experiment  
params=[True,True,True]#dropout,data augmentation,bacth norm  
batch_size=32  
lr=1e-10  
decay=0  #from adam
epochs=50

```
part2:
The experiment's names goes as follow:
  - 2_1:Baseline model 
  - 2_2:extra test set -> in this cases we need to comment the line to fit the model (and the ones to plot the results) and move the new test set to the test folder 

```python
experiment='2_2' #name of the experiment 
batch_size=64  
lr=1e-10  
decay=0  #from adam
epochs=200

```
