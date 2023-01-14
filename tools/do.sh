#!/bin/bash

set -e

#/WaymoOpenDataset/training/training_training_0026.tar
#/WaymoOpenDataset/training/training_training_0000.tar
#/WaymoOpenDataset/training/training_training_0024.tar
#/WaymoOpenDataset/training/training_training_0004.tar
#/WaymoOpenDataset/training/training_training_0018.tar
#/WaymoOpenDataset/training/training_training_0009.tar
packages=" 
/WaymoOpenDataset/training/training_training_0021.tar
/WaymoOpenDataset/training/training_training_0007.tar
/WaymoOpenDataset/training/training_training_0030.tar
/WaymoOpenDataset/training/training_training_0015.tar
/WaymoOpenDataset/training/training_training_0022.tar
/WaymoOpenDataset/training/training_training_0008.tar
/WaymoOpenDataset/training/training_training_0020.tar
/WaymoOpenDataset/training/training_training_0006.tar
/WaymoOpenDataset/training/training_training_0027.tar
/WaymoOpenDataset/training/training_training_0005.tar
/WaymoOpenDataset/training/training_training_0016.tar
/WaymoOpenDataset/training/training_training_0019.tar
/WaymoOpenDataset/training/training_training_0003.tar
/WaymoOpenDataset/training/training_training_0002.tar
/WaymoOpenDataset/training/training_training_0012.tar
/WaymoOpenDataset/training/training_training_0029.tar
/WaymoOpenDataset/training/training_training_0013.tar
/WaymoOpenDataset/training/training_training_0001.tar
/WaymoOpenDataset/training/training_training_0023.tar
/WaymoOpenDataset/training/training_training_0011.tar
/WaymoOpenDataset/training/training_training_0031.tar
/WaymoOpenDataset/training/training_training_0010.tar
/WaymoOpenDataset/training/training_training_0014.tar
/WaymoOpenDataset/training/training_training_0028.tar
/WaymoOpenDataset/training/training_training_0017.tar
/WaymoOpenDataset/training/training_training_0025.tar
/WaymoOpenDataset/validation/validation_validation_0003.tar
/WaymoOpenDataset/validation/validation_validation_0005.tar
/WaymoOpenDataset/validation/validation_validation_0002.tar
/WaymoOpenDataset/validation/validation_validation_0001.tar
/WaymoOpenDataset/validation/validation_validation_0000.tar
/WaymoOpenDataset/validation/validation_validation_0007.tar
/WaymoOpenDataset/validation/validation_validation_0006.tar
/WaymoOpenDataset/validation/validation_validation_0004.tar"


#packages=" 
#/WaymoOpenDataset/training/training_training_0026.tar
#"


#echo $packages
for var in ${packages[@]}
do


   file=${var##*/}

   echo /root/autodl-tmp${var}
   
   #1. extract tar file
   tar xf /root/autodl-tmp${var}  -C /root/autodl-tmp/DataSet/

   #2. remove tar file
   rm -rf /root/autodl-tmp${var} 

   #3. extract json and image
   python project.py  --tfrecord_dir /root/autodl-tmp/DataSet

   #4. remove tfrecord
   rm -rf /root/autodl-tmp/DataSet/*

#   #5. package 
#   tar cf /root/autodl-tmp/out/${file} -C /root/autodl-tmp/ extend
#
#   #6. remove extend.
#   rm -rf /root/autodl-tmp/extend
#
done




