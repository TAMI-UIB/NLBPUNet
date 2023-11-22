# NLRNET
If you use our code, please cite the following papers.

D. Lei, H. Chen, L. Zhang and W. Li, "NLRNet: An Efficient Nonlocal Attention ResNet for Pansharpening," in IEEE Transactions on Geoscience and Remote Sensing, vol. 60, pp. 1-13, 2022, Art no. 5401113, doi: 10.1109/TGRS.2021.3067097.

解压后源码包主要目录结构及文件作用。

├──NLRNET 

│    ├──experiment			  //抽离出的各种类型网络中通用的一些方法函数

│    │    ├──__init__.py                  //模块导入初始化

│    │    ├──base_experiment.py           //加载数据、训练、输出模型的通用方法

│    │    ├──cnn_experiment.py            //扩展base_experiment，自定义损失函数

│    │    ├──gan_experiment.py            //针对GAN网络训练的配置

│    ├──model                             //输出模型保存文件夹

│    ├──net                           

│    │    ├──nlrnet                 

│    │    │    ├──GCblock.py           //non-local注意力机制模块

│    │    │    ├──NLRNet.py            //整体网络结构模型

│    ├──processing                     

│    │    ├──generate_dataset.py       //划分数据集，生成训练集、测试集

│    │    ├──resource_manager.py       //对数据集进行预处理

│    ├──resource                       //数据集存放文件夹

│    ├──cache                        //该文件夹下保存测试集生成的img.pickle

│    ├──utils    //一些方法函数，其中的process_img_pickle.py 用于生成结果的.mat文件

│    │    ├──__init__.py                

│    │    ├──array2raster.py          //array数组转为光栅

│    │    ├──assessment.py            //融合结果指标测试函数

│    │    ├──process_gf_pickle.py     //对4通道数据集进转换处理

│    │    ├──process_wv_pickle.py     //对8通道数据集进转换处理

│    │    ├──recorder.py              //网络训练打印输出日志

│    │    ├──tools.py                 //指标测试代码
