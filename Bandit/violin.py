import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
"""
k=10
value_array = np.random.normal(0,1,k)
print(value_array)

reward = []
action = []
for i in range(k):
    value = value_array[i]
    reward.append(np.random.normal(value,10,100000))
    action.append(np.ones(100000)*i)
reward = np.array(reward).reshape(1, -1)[0]
action = np.array(action).reshape(1, -1)[0]

dataframe = pd.DataFrame({'reward':reward,'action':action})

dataframe.to_csv("test.csv",index=False,sep=',')

"""
r_action = pd.read_csv('test.csv')
sns.set(style='whitegrid', color_codes=True)
sns.violinplot(x='action', y='reward',data=r_action)
plt.show()
