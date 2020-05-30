This project consist in implementing an environnement with caracteristics :
1. This is a typical grid world, with 4 stochastic actions. The actions might result in movement in a direction other than the one intended with a probability of 0.1. Otherwise it will transition to one of the other neighbouring cells with probability 0.1/3.
2. Transitions that take you off the grid will not result in any change.
3. There is also a gentle Westerly blowing, that will push you one additional cell to the east, with a probability of 0.5.
4. The episodes start in one the start states in the first column, with equal probability.
5. There are three variants of the problem, A, B, and C, in each of which the goal is in the square marked with the respective alphabet. There is a reward of +10 on reaching the goal.
6. There is a puddle in the middle of the gridworld which the agent would like to avoid. Every transition into a puddle cell gives a negative reward depending on the depth of the puddle at that point, as indicated in the figure.
You can find the environnement in puddle_world.png

And then to execute Sarsa and Q_leraning on the environnement, my environnement is implemented in gym-foo. To launch it, with sarsa or q-learning, you can just go in the code and change the algorithm parameter to "Sarsa" or "Q-Learning". To launch the code, you just need to typein the command prompt python Start.py.
You can find in the graph, the plots for all the different parameters.
