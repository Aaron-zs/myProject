# 数据文件说明
## x_dark_val_area.npy 
特征信息文件，包括2800+个样本，每个样本40个维度(float)[40]。<br/>
前20个维度：(float)[20],每个样本在20帧时序上，正常与异常部分吸光的差值<br/>
后20个维度：(float)[20],每个样本在20帧时序上，吸光部分的面积差值<br/>
## y_dark_val_area.npy
分类标签文件，包括2800+个样本，每个样本2个维度(int)[2]。<br/>
对于每个样本：正常[1,0]；异常[0,1]。
## names_dark_val_area.npy / names_dark_light_val.npy
样本名称文件，包括2800+个样本，每个样本一个字符串(string)。<br/>
## x_dark_val_area.npy 
特征信息文件，包括2800+个样本，每个样本40个维度(float)[40]。<br/>
前20个维度：(float)[20],每个样本在20帧时序上，正常与异常部分吸光的差值<br/>
后20个维度：(float)[20],每个样本在20帧时序上，正常与异常部分透光的差值<br/>
## y_dark_val_area.npy
分类标签文件，包括2800+个样本，每个样本2个维度(int)[2]。<br/>
对于每个样本：正常[1,0]；异常[0,1]。

## 其他
1、三个文件的顺序严格对齐。即，x文件中第一个样本特征，对应y文件中第一个样本分类，对应name文件中第一个样本名称。
2、npy格式文件是python中numpy包自带存储格式，使用data=np.load("./name.npy") 即可读取。