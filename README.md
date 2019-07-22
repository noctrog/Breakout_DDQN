# DDQN: Dueling Deep Q Network (for Atari Breakout)

This project consists of a Dueling DQN model that learns to play Breakout.

For training, the model is fed with a prioritized buffer, so it can learn faster from its mistakes.
Moreover, noisy nets are used instead of epsilon-greedy, as it makes the classifier automatically
adjust for exploration thus reducing hyperparameter count.

![](https://media.giphy.com/media/jt8L43CbwifMoocXTV/giphy.gif)

## Params

### dqn_train.py

 * **--env**: change gym environment
 * **--dueling**: enables Dueling DQN
 * **--double**: enables Double DQN (can be combined with Dueling DQN)
 * **--cuda**: use CUDA for training and playing
 * **--reward**: stop when mean reward reaches the value given
 * **--clip**: clip gradient to given value
 * **--play_steps**: play *n* times more steps and increase *n* times batch size. Used to increase
     training speed.

### play.py

 * **--cuda**: play using cuda (runs much faster)
 * **--record**: save play recording in *recordings* folder
