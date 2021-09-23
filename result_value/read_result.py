import result_value.value as gl
import pandas as pd
import matplotlib.pyplot as plt

def print_tau(df):
    '''
    绘制本地聚合频次的变化情况，探索其中的规律，
    比如与准确率的关系，
    或者与数据分布的关系
    :param df: 存放的数据
    :return:
    '''
    print(df.head())
    X = df["COM_TIMES"]
    Y = df["tau"]
    plt.figure(figsize=(8, 4))
    plt.plot(X, Y, label="$sin(x)$", color="blue", linewidth=2)
    plt.xlabel("COM_TIMES")
    plt.ylabel("tau")
    plt.title("aggregation times and tau")
    plt.show()

def print_loss(df):
    '''
    绘制本地聚合频次的变化情况，探索其中的规律，
    比如与准确率的关系，
    或者与数据分布的关系
    :param df: 存放的数据
    :return:
    '''
    print(df.head())
    X = df["COM_TIMES"]
    Y = df["loss"]
    plt.figure(figsize=(8, 4))
    plt.plot(X, Y, label="$sin(x)$", color="red", linewidth=2)
    plt.xlabel("COM_TIMES")
    plt.ylabel("tau")
    plt.title("aggregation times and loss")
    plt.show()

df=pd.read_csv(gl.PATH)
#print_tau(df)
print_loss(df)

