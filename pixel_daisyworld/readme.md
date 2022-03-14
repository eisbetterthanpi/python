
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eisbetterthanpi/python/pixel_daisyworld/blob/master/pixel_daisyworld.ipynb)

# pixel daisyworld
Pixel daisyworld is an extension of Daisyworld simulation, a model of an imaginary planet
Lovelock and Andrew Watson to support the Gaia theory

# Mastering Atari with Discrete World Models

Implementation of the [pixel][website] yaya

<p align="center">
<!-- <img width="90%" src="https://imgur.com/gO1rvEn.gif"> -->
</p>

![Alt Text](data/Awb400.gif)
![Alt Text](data/Tl400.gif)


If you find this code useful, please credit `eisbetterthanpi`

[website]: https://github.com/eisbetterthanpi


## Using the Package

The easiest way to run DreamerV2 on new environments is to install the package
via `pip3 install dreamerv2`. The code automatically detects whether the
environment uses discrete or continuous actions. Here is a usage example that
trains DreamerV2 on the MiniGrid environment:

```python
import gym
import gym_minigri
env = gym_minigrid.wrappers.RGBImgPartialObsWrapper(env)
dv2.train(env, config)
```

## Manual Instructions

Gaia theory
Actively regulated
Atmosphere constant
Destabalisation life environment interactions lead to local reset
How much life affect the environment vs natural processes

https://www.brookes.ac.uk/geoversity/publications/an-analysis-of-the-impact-of-the-gaia-theory-on-ecology-and-evolutionary-theory
increased biodiversity and increasing stability of populations


#### [Open `pixel_daisyworld.ipynb` in Google Colab](pixel_daisyworld.ipynb)


## Observations
At high luminosities,
Large patches of black die and get splitup
Black surrounded by white survives
White takeover at high temp	but black dun at low prob bec unsymmetric albedo
Great heat death is due to expansion of ground cover
Black starts dying from the center of large patches of black


## Future work
Importance of atmosphere in the simulation
- Ocean conveyor belt disruption [mixed precision][mixed] guide. You can disable mixed precision by passing
`--precision 32` to the training script. Mixed precision is faster but can in
principle cause numerical instabilities.

[mixed]: https://www.tensorflow.org/guide/mixed_precision

