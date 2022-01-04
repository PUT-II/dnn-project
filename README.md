# Testing Upside-Down Reinforcement Learning in Super Mario Bros environment

This repository contains project created for DNN course at Poznan University of Technology. Main goal of this project is
to implement and test Upside-Down Reinforcement Learning in Super Mario Bros.

## Project content

### UDRL

`UDRL` class handles UDRL training. `train` is the most important method of this class.
**TODO: Some of its methods could be moved to some `UdrlAgent` class**

### Behavior

`Behavior` class handles action prediction. In this project combined state, command and info (mario position + mario
status) is used for prediction.

### Replay buffer

`ReplayBuffer` class stores episodes and provides methods for data access.
**TODO: Could inherit directly from `list`**

###

## Current state of project

* Refactored UDRL project for Lunar Lander and repurposed it for Super Mario Bros
* Implemented CNN for feature extraction from state (image)
* Implemented state pre-processing (grayscale, 128x128 rescale)
* Created train and test scripts
* Agent if not learning

## Authors

* Tomasz Kiljańczyk
* Rafał Ewiak

## Sources and inspirations

* [Training Agents using Upside-Down Reinforcement Learning](https://arxiv.org/abs/1912.02877)
* [Demystifying Upside-Down Reinforcement Learning (a.k.a ⅂ꓤ)](https://medium.com/@jscriptcoder/demystifying-upside-down-reinforcement-learning-a-k-a-%EA%93%A4-b7bd4214b33f)
* [A Simple Guide To Reinforcement Learning With The Super Mario Bros. Environment](https://medium.com/geekculture/a-simple-guide-to-reinforcement-learning-with-the-super-mario-bros-environment-495a13974a54)