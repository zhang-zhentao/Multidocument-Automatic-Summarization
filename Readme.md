# 多文档自动摘要
## 基本信息
编写人：2020302419张镇韬
编写时间：2023年5月3日
## 问题描述

## 文档结构
### DUC04
Document Understanding Conference是由美国国家标准与技术研究所（National Institute of Standards and Technology， NIST）组织的旨在促进文本摘要自动化（Automatic Text Summarization ）技术发展的年会。   
我们采用DUC2004任务2的数据集作为实验的数据集。该数据集包含待摘要的文档集以及每个主题类文档的专家摘要。DUC数据集包含50个主题类，每个主题下包含10篇文档。对于每一个主题类的文档集，数据集提供了4个不同的专家摘要作为评判标准。

**model**

04model文件夹下包含200个项目，即为50个主题的专家摘要（每一个主题4篇）

**participating systems result**

2文件夹下包含1949个项目，即为50个主题的专家摘要和比赛摘要（每一个主题4篇专家摘要和35篇队伍摘要，但19号队伍的D31001主题摘要缺失）

**unpreprocess data**

docs文件夹下包含五十个文件夹，每个文件夹内包含十篇文档，即为原始数据。
### Method
本文件夹包含了我们实现的各种方法

**Code**

Code文件夹下包含我们实现的四种方法（Baseline、TFIDF、Cluster、Sentence2Vec）以及ROUGH评价算法（Evaluate），还有利用这个评价算法对队伍进行排序（Rank）

**distilbert-base-nli-mean-tokens**

此为我们的Sentence2Vec模型文件夹，是下载的训练好的模型

**Result**

Eesult文件夹下为四个文件夹，分别是Code文件夹中四个方法的运行结果。rank.xlsx文件是Rank算法的运行结果，即35个队伍的排序表格