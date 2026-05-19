# MDP-Snake-Game
Simple snake game with Markov Decision Process reinforcement learning solver. Created using PyGame and GeeksforGeeks MDP implementation: https://www.geeksforgeeks.org/machine-learning/what-is-reinforcement-learning/.

# Demo Video:
<p align="center">
  <img src="Demo-GIF.gif"/>
</p>

# Features:
## PyGame:
- snake game
- get apples without going out of bounds or colliding with your own body
- controls: 'w' for up, 's' for down, 'a' for left, 'd' for right
- note: opposite direction controls are not allowed e.g: if travelling up, 's' to go down is not allowed in the code
## Markov Decision Process Solver:
- reinforcement learning navigator
- simulates a replica environment
- rewards: +50 for going closer to or reaching apple, -10 for going out of bounds, crashing into yourself or going farther from apple, -30 for going closer to your own body, -1 for each step (avoid unnecessary steps)
- number of navigation attempts: 100 (tunable)
- max steps solver can take: 10 (first action is used, rest is discarded)

# Quick Start:
1. Install dependencies:
```
pip install numpy
pip install pygame
```
2. run game.py and enjoy

# Additional Notes:
I did not create nor design the Markov Decision Process implementation. It was taken from Geeksforgeeks reinforcement learning page and modified to work with this game rather than a maze navigation goal.

Snake game was created by me using PyGame.