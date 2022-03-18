一些数据的特征
Bank:
label(-1)
age(0)

Adult:
label(-1)
age(0), race(6) and gender(7)

German:
label(0)
gender(6) and age(9)
这里面的gender并不仅仅是性别
个人婚姻状况和性别（A91：男性：离婚/分居，
A92：女性：离婚/分居/已婚，
A93：男性：单身，
A94：男性：已婚/丧偶，A95：女性：单身）


Compas:
FN 似乎只有一个 保护属性？
label(-1)
race(2)
sex(0)应该也是保护属性但是作为一个犯罪数据集,sex 并没有当作 保护属性
需要注意的是 预测的 scores 其实 scores_text 就是天生的一个分箱,只是删除了而已
1,2,3,4 low
5,6,7 medium
8,9,10 high
对于数据集的 Label 是要分成二分类
10分类, 30%左右
3分类, 60 左右
2分类 , 70 左右

没有搞明白 FN 到底是怎么分类的