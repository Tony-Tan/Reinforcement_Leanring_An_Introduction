import gym

env = gym.make("Blackjack-v0")
for i in range(1000):
    t=env.reset()
    if t[0]<6:
        print(t)