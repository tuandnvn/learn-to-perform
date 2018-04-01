# Teaching simulated agents to perform complex spatial-temporal activities

## Abstract

we introduce a framework in which computers learn to enact complex temporal-spatial actions by observing humans. Our framework processes  motion capture data of human subjects performing actions, and uses qualitative spatial reasoning to learn multi-level representations for
these actions. Using reinforcement learning, these observed sequences are used to guide a simulated agent to perform novel actions. To evaluate, we visualize the action being performed in an embodied 3D simulation environment, which allows evaluators to judge whether the system has successfully learned the novel concepts. This approach complements other planning approaches in robotics and demonstrates a method of teaching a robotic or virtual agent to understand predicate-level distinctions in novel concepts.

## Content

This is a project to demonstrate the idea that we can teach robots to execute certain complex actions by allowing them to observe human experts to perform the same actions a number of times, and using planning with reinforcement learning methods to simulate the action. While the method might be error-proned because of the number of small demonstrations, and the lacks of other competitive actions or negative samples of actions, we proved that 

This is our paper at [AAAI Spring Symposium 2018](https://www.researchgate.net/profile/Tuan_Do14/publication/322836314_Teaching_Virtual_Agents_to_Perform_Complex_Spatial-Temporal_Activities/links/5a724463458515512075e396/Teaching-Virtual-Agents-to-Perform-Complex-Spatial-Temporal-Activities.pdf). 

You can find our presentation at AAAI Spring Symposium 2018 [here](miscellanous/AAAI-SS-2018-jp-edits.pptx).

## Install

My code requires installation of tensorflow, and optionally keras. I use Anaconda python 3.6.2 distribution. Installation of tensorflow can be found [here](https://www.tensorflow.org/install/). Other required library could be installed as followings:

```
conda install -c anaconda html5lib

pip install gym==0.9.4

pip install matplotlib

pip install scipy

pip install pandas
```

## Try it

You can try the python notebook at [slide_around_demo](demo/slide_around_demo.ipynb). This notebook demonstrate the difficulty in learning a complex temporal action like *Slide Around* by just observing positive demonstrations of the action. 

The python notebook at [slide_around_demo_with_feedback](demo/slide_around_demo_with_feedback.ipynb). This notebook loads 30 different setups with generated demonstrations from the first method, and it also loades 30 scores given by two annotators. It then retrained the learning model, and perform the action with better result.  

