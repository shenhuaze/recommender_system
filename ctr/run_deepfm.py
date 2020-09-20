
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from deepctr.models import DeepFM
from deepctr.feature_column import SparseFeat, DenseFeat,get_feature_names

if __name__ == '__main__':
    # 加载数据集
    data = pd.read_csv('../dataset/criteo_sample.txt')

    # target字段名
    target = ['label']

    # 稠密特征字段名列表
    dense_features = ['I' + str(i) for i in range(1, 14)]
    # 缺失值填充为0
    data[dense_features] = data[dense_features].fillna(0)

    # 稀疏类别特征字段名列表
    sparse_features = ['C' + str(i) for i in range(1, 27)]
    # 确实值填充为"-1"
    data[sparse_features] = data[sparse_features].fillna('-1')

    # 对每列稀疏特征，都转化为label encoding，也就是将每个特征值转化为对应的特征值index，一列 -> 一列
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])

    # 将特征值归一化到(0, 1)区间内
    mms = MinMaxScaler(feature_range=(0, 1))
    # 对每列稠密特征的值做归一化，一列 -> 一列
    data[dense_features] = mms.fit_transform(data[dense_features])

    # 针对每一列稀疏特征生成一个SparseFeat对象，每个SparseFeat对象都有一个特征名、vocab_size、维度
    sparse_columns = [SparseFeat(feat, vocabulary_size=data[feat].nunique(), embedding_dim=4) for feat in sparse_features]
    # 针对每一列稠密特征生成一个DenseFeat对象
    dense_columns = [DenseFeat(feat, 1) for feat in dense_features]
    # 将稀疏特征对象列表和稠密特征对象列表拼接到一起
    fixlen_feature_columns = sparse_columns + dense_columns

    # 得到线性部分的特征对象列表
    linear_feature_columns = fixlen_feature_columns
    # 得到dnn部分的特征对象列表
    dnn_feature_columns = fixlen_feature_columns
    # 将线性部分的特征对象列表与dnn部分的特征对象列表拼接到一起
    combined_feature_columns = linear_feature_columns + dnn_feature_columns
    # 根据拼接之后的特征对象列表，得到所有distinct的特征名
    feature_names = get_feature_names(combined_feature_columns)

    # 切分训练集和测试集
    train, test = train_test_split(data, test_size=0.2)

    # 得到训练集输入，每个特征名以及对应的所有训练样本的特征值
    train_model_input = {name: train[name].values for name in feature_names}
    # 得到测试集输入，每个特征名以及对应的所有测试样本的特征值
    test_model_input = {name: test[name].values for name in feature_names}

    # 初始化DeepFM模型，传入的参数分别为：线性部分的特征对象列表linear_feature_columns，dnn部分的特征对象列表dnn_feature_columns
    # 和任务task，这里是二分类任务binary
    model = DeepFM(linear_feature_columns=linear_feature_columns,
                   dnn_feature_columns=dnn_feature_columns,
                   task='binary')

    # 添加模型训练的参数：optimizer为adam，损失函数为binary_crossentropy，metrics为binary_crossentropy
    model.compile(optimizer="adam",
                  loss="binary_crossentropy",
                  metrics=['binary_crossentropy'])

    # 在训练集上训练模型，这里的x感觉是用来填充placeholder的，或者说填充Input对象的
    history = model.fit(x=train_model_input,
                        y=train[target].values,
                        batch_size=256,
                        epochs=10,
                        verbose=2,
                        validation_split=0.2)

    # 在测试集上批量预测
    pred_ans = model.predict(x=test_model_input,
                             batch_size=256)
