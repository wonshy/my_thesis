#!/bin/bash
pip install tqdm
pip install ortools==9.2.9972
pip install efficientnet_pytorch==0.7.0
pip install setuptools==58.2.0
pip install opencv_python==4.2.0.32
pip install tensorboardX
pip install thop
pip install scipy
sudo apt-get install libglib2.0-0 -y

cd ../model/nms/
#python setup.py install
python3 setup.py install --user
cd -
