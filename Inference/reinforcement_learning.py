# Reinforcement Learning
# Just a rough idea implementation
# Hasn't been executed
# Just for the intuition
obs = env.reset()
h = mdnrnn.initial_state

# Done variable, if game is over
done = False

# To maximize the cumulative reward by finding optimal parameter of the contollers in matrix Wc
cumulative_reward = 0

while not done:
    # From CNN VAE
    z = cnnvae(obs)
    # Action from controller
    a = controller((z, h))
    # Calling the environment and step function from the game
    obs, reward, done = env.step()
    # Incrementing the cum reward by the reward obtained
    cumulative_reward += reward
    #taking back the hidden state(or update it)
    h = mdnrnn([a,z,h])