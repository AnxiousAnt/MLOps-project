import pandas as pd

test = pd.read_csv('Dataset/test.csv')
train = pd.read_csv('Dataset/train.csv')

print(test.info())
print(train.info())