# Teaching simulated agents to perform complex spatial-temporal activities

## Abstract

we introduce a framework in which computers learn to enact complex temporal-spatial actions by observing humans. Our framework processes  motion capture data of human subjects performing actions, and uses qualitative spatial reasoning to learn multi-level representations for
these actions. Using reinforcement learning, these observed sequences are used to guide a simulated agent to perform novel actions. To evaluate, we visualize the action being performed in an embodied 3D simulation environment, which allows evaluators to judge whether the system has successfully learned the novel concepts. This approach complements other planning approaches in robotics and demonstrates a method of teaching a robotic or virtual agent to understand predicate-level distinctions in novel concepts.

## Content

This is a project to demonstrate the idea that we can teach robots to execute certain complex actions by allowing them to observe human experts to perform the same actions a number of times, and using planning with reinforcement learning to simulate the action.

This is our position paper submitted to AAAI Spring Symposium 2018.

https://github.com/tuandnvn/learn-to-perform/blob/master/miscellanous/Tuan_AAAI_2018%20November%204.pdf

## Install

My code requires installation of tensorflow, and optionally keras. I use Anaconda python 3.6.2 distribution. Installation of tensorflow can be found [https://www.tensorflow.org/install/](here). Other required library could be installed as followings:

'''
conda install -c anaconda html5lib

pip install gym==0.9.4

pip install matplotlib

pip install scipy

pip install pandas
'''

## Try it 

