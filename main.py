from emotion import Psychologist
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
if __name__ == '__main__':
    psy = Psychologist(50000)
    test_ori = psy.orign('comment.csv','utf-8','comment')

    # test_ori=['崽崽是要把微博当朋友圈了吗？但我没意见啊哈哈哈哈哈多来点多来点[舔屏][舔屏]']

    _, test_pad = psy.data_process(test_ori)
    # accuracy 96%
    model = psy.load_model('psy_model.keras')
    predictions = model.predict(test_pad)

    # print(predictions)

    results=[]
    for prob in predictions:
        if prob > 0.9:
            label=1
        elif prob < 0.1:
            label=-1
        else:
            label=0
        results.append(label)
    results = np.array(results)
    positive_rate = np.sum(results == 1)/len(results)
    neutral_rate = np.sum(results == 0)/len(results)
    negative_rate = np.sum(results == -1)/len(results)

    print(f'积极比例: {positive_rate:.2f}, 中性比例:{neutral_rate:.2f}, 消极比例: {negative_rate:.2f}')

    font_path = '微软雅黑.ttf'
    font_manager.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = font_manager.FontProperties(fname=font_path).get_name()
    # 绘制饼图
    labels = ['积极', '中性', '消极']
    sizes = [positive_rate, neutral_rate, negative_rate]
    colors = ['#66b3ff', '#99ff99', '#ff9999']  # 蓝、绿、红
    # explode = (0.05, 0, 0.05)  # 突出显示积极和消极部分

    plt.figure(figsize=(8, 6))
    plt.pie(sizes, labels=labels, colors=colors,
            autopct='%1.1f%%', shadow=True, startangle=90)
    plt.axis('equal')
    plt.title('翟天临评论区情感分布', fontsize=15)
    plt.show()
