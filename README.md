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
```python
experiment='1_5' #name of the experiment  
params=[True,True,True]#dropout,data augmentation,bacth norm  
batch_size=32  
lr=1e-10  
decay=0  #from adam
epochs=50

```
part2:
```python
experiment='2_2' #name of the experiment 
batch_size=64  
lr=1e-10  
decay=0  #from adam
epochs=200

```
