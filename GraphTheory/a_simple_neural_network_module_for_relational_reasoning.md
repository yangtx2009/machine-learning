# [A simple neural network module for relational reasoning (NIPS 2017)](https://drive.google.com/file/d/1pPa0DBFI0EaOY_3IhsBovwteweRpHf1G/view?usp=drivesdk)


## Authors

- Adam Santoro
- David Raposo
- David GT Barrett
- Mateusz Malinowski
- Razvan Pascanu
- Peter Battaglia
- Timothy Lillicrap.

## Overview

相当于构建一个全连有向图网络,一个网络预测所有连接,并将所有edge相加,输出一个path上的结果
- 使用CNN将图像处理成objects (k个dxd feature map加上其坐标)
- 提问用LSTM提取question embedding q
- 再用4层256全连网络(gθ)对每个问题预测出固定长度relation features,并点对点相加得出一个relation feature
- 3层全量预测出answer的softmax结果并最小化loss

