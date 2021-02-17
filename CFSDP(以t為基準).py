#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/3/26 16:56
# @Author  : jyl
# @File    : CFSDP.py
import numpy as np
import matplotlib.pyplot as plt
import collections
from sklearn import metrics
from sklearn.metrics import davies_bouldin_score
import pandas as pd
# 例v=np.array([[7,8],[5,2],[2,3],[6,4]])
# 计算任意两点之间的欧氏距离,并存储为矩阵
def caldistance(v):
    distance = np.zeros(shape=(len(v), len(v)))
    for i in range(len(v)):
        for j in range(len(v)):
            if i > j:
                distance[i][j] = distance[j][i]
            elif i < j:
                distance[i][j] = np.sqrt(np.sum(np.power(v[i] - v[j], 2)))
    return distance


# 选出dc,t值表示平均每个点周围距离最近的点数为总点数的t%,根据所给的t选择出dc
# t的输入为百分数的整数部分,如2%只需要输入2即可
def chose_dc(dis, t):
    temp = []
    print('len dis',len(dis[0]))
    for i in range(len(dis[0])):
        for j in range(i + 1, len(dis[0])):
            temp.append(dis[i][j])
    print('dis shape',dis.shape)
    #  升序排列
    temp.sort()
    print('temp: ',temp)
    print('總距離個數',len(temp))
    dc = temp[int(len(temp) * t / 100)]
    print('dc: ',dc)
    return dc

#  通过数距离小于dc的点的个数来衡量一个点的密度（离散型）**
def count_density(distance, dc):
    density = np.zeros(shape=len(distance))
    for index, node in enumerate(distance):
        density[index] = len(node[node < dc])
    return density


#  通过公式np.sum(np.exp(-(node / dc) ** 2))衡量一个店的密度（连续型）**
#node=dij
def continous_density(distance, dc):
    density = np.zeros(shape=len(distance))
    for index, node in enumerate(distance):
        density[index] = np.sum(np.exp(-(node / dc) ** 2))
        print(index,':',density[index])
    return density


# 计算密度大于自身的点中距离自己最近的距离以及该点的直属上级
def node_detal(density, distance):
    detal_ls = np.zeros(shape=len(distance))
    closest_leader = np.zeros(shape=len(distance), dtype=np.int32)
    delta = 11
    d_theta = 0.1
    for index, node in enumerate(distance):
        theta = density[index]
        Stop = False
        while Stop != True:
            #  点密度大于当前点的点集合（一维数组）
            density_larger_than_node = np.squeeze(np.argwhere(density > theta))
            #  存在密度大于自己的点
            if density_larger_than_node.size != 0:
                #  所有密度大于自己的点与自己的距离集合（一维数组或者一个数）
                distance_between_larger_node = distance[index][density_larger_than_node]
                # print(index,distance_between_larger_node)
                detal_ls[index] = np.min(distance_between_larger_node)
                if detal_ls[index] >= delta:
                    theta = theta - d_theta
                else:
                    min_distance_index = np.squeeze(np.argwhere(distance_between_larger_node == detal_ls[index]))
                    #  存在多个密度大于自己且距离自己最近的点时，选择第一个点作为直属上级
                    if min_distance_index.size >= 2:
                        min_distance_index = np.random.choice(min_distance_index)
                    if distance_between_larger_node.size > 1:
                        closest_leader[index] = density_larger_than_node[min_distance_index]
                    else:
                        closest_leader[index] = density_larger_than_node
                    Stop = True
            #  对于最大密度的点
            else:
                detal_ls[index] = np.max(distance)
                closest_leader[index] = index
                Stop = True
    return detal_ls, closest_leader


# 确定类别点,计算每点的密度值与最小距离值的乘积，并画出决策图，以供选择将数据共分为几个类别
def show_nodes_for_chosing_mainly_leaders(density, detal_ls):
    #  由于密度和最短距离两个属性的数量级可能不一样，分别对两者做归一化使结果更平滑
    normal_den = (density - np.min(density)) / (np.max(density) - np.min(density))
    normal_det = (detal_ls - np.min(detal_ls)) / (np.max(detal_ls) - np.min(detal_ls))
    gamma = normal_den * normal_det
    # plt.figure(num=2, figsize=(15, 10))
    # plt.scatter(x=range(len(detal_ls)), y=-np.sort(-gamma), c='k', marker='o', s=-np.sort(-gamma) * 100)
    # plt.xlabel('data_num')
    # plt.ylabel('gamma')
    # plt.title('Guarantee The Leader')
    # plt.show()
    return gamma


# 确定每点的最终分类
def clustering(closest_leader, chose_list):
    for i in range(len(closest_leader)):
            print('點:', i)
            while closest_leader[i] not in chose_list:
                j = closest_leader[i]
                print(j)
                closest_leader[i] = closest_leader[j]
                print('closest_leader[i]',closest_leader[i])
    new_class = closest_leader[:]
    return new_class  # new_class[i]表示第i点所属最终分类


def show_result(new_class, norm_data, chose_list):
    colors = [
        '#FF0000', '#FFA500', '#FFFF00', '#00FF00', '#228B22',
        '#0000FF', '#FF1493', '#EE82EE', '#000000', '#FFA500',
        '#00FF00', '#006400', '#00FFFF', '#0000FF', '#FFFACD',
        '#770077', '#008866', '#000088', '#9F88FF','#3A0088',
        '#660077', '#FF00FF','#0066FF', '#00FF00', '#7744FF',
        '#33FFDD', '#CC6600', '#886600', '#227700', '#008888',
        '#FFFF77', '#D1BBFF'
              ]
    # 画最终聚类效果图
    leader_color = {}
    main_leaders = dict(collections.Counter(new_class)).keys()
    for index, i in enumerate(main_leaders):
        leader_color[i] = index
    plt.figure(num=3, figsize=(15, 15))
    for node, class_ in enumerate(new_class):
        #  标出每一类的聚类中心点
        if node in chose_list:
            plt.scatter(x=norm_data[node, 0], y=norm_data[node, 1], marker='*', s=500, c='black',alpha=1)
        else:
            plt.scatter(x=norm_data[node, 0], y=norm_data[node, 1], c=colors[leader_color[class_]], s=100, marker='o',alpha=1)
    plt.title('The Result Of Cluster')
    plt.show()


# 画detal图和原始数据图
def show_optionmal(den, det, v):
    plt.figure(num=1, figsize=(15, 9))
    ax1 = plt.subplot(121)
    for i in range(len(v)):
        plt.scatter(x=den[i], y=det[i], c='k', marker='o', s=15)
    plt.xlabel('Density')
    plt.ylabel('Detal')
    plt.title('Chose Leader')
    plt.sca(ax1)

    ax2 = plt.subplot(122)
    for j in range(len(v)):
        plt.scatter(x=v[j, 0], y=v[j, 1], marker='o', c='k', s=15)
    plt.xlabel('axis_x')
    plt.ylabel('axis_y')
    plt.title('Dataset')
    plt.sca(ax2)
    plt.show()


def main(input_x):
    # a =int(input('input start clusters num'))
    # b =int(input('input final clusters num'))
    a=2
    b=2
    num_list = []
    list = []
    for leaders_num in range(a,(b+1)):
        for t in range(59,100):
            print('t=',t)
            norm_data = input_x
            distance = caldistance(norm_data)  # 制作任意两点之间的距离矩阵
            dc = chose_dc(distance, t)  # 根据t选择合适的dc
            density = continous_density(distance, dc)  # 统计每点的密度
            detal_ls, closest_leader = node_detal(density, distance)  # 统计每点的直接上司
            # show_optionmal(density, detal_ls, norm_data)  # 展示原始数据集并为选择司令个数提供参考
            scores = show_nodes_for_chosing_mainly_leaders(density, detal_ls)  # 进一步确认选择的司令个数是否正确
            chose_list = np.argsort(-scores)[: leaders_num]  # 选择
            print(chose_list)
            new_class = clustering(closest_leader, chose_list)  # 确定各点的最终归属（哪个司令）
            show_result(new_class, norm_data, chose_list)  # 展示结果
            num_list += [str(leaders_num)]
            calinski = metrics.calinski_harabaz_score(norm_data, new_class)
            davies = davies_bouldin_score(norm_data, new_class)
            silhouette = metrics.silhouette_score(norm_data, new_class, metric='euclidean')
            list.append([calinski, davies, silhouette])

    name = ['calinski_harabaz_score', 'davies_bouldin_score', 'silhouette_score']
    df = pd.DataFrame(list, index = num_list, columns = name)
    print(df)




if __name__ == '__main__':
    data = r'C:\Users\Larrywu\Desktop\schoolHW\CFSDP\Clustering-Python-master2\annulus.txt'
    raw_data = np.loadtxt(data, delimiter='	', usecols=[0, 1])
    main(raw_data)






