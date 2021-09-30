
from operator import index
import pandas as pd
from sklearn.utils import shuffle

file = r"C:\Users\nvtru\Desktop\auto-recommender\ml-1m\ratings.dat"


df = pd.read_csv(file,sep="::", names=["u_id","i_id","rating","timestamps"])
df = shuffle(df)
train = df.sample(frac=0.8, random_state=5)
val = df.drop(train.index.tolist()).sample(frac=0.5, random_state=8)
test = df.drop(train.index.tolist()).drop(val.index.tolist())
train.to_csv(r"C:\Users\nvtru\Desktop\auto-recommender\ml-1m\train.txt",index= False)
test.to_csv(r"C:\Users\nvtru\Desktop\auto-recommender\ml-1m\test.txt",index= False)
val.to_csv(r"C:\Users\nvtru\Desktop\auto-recommender\ml-1m\val.txt",index= False)
