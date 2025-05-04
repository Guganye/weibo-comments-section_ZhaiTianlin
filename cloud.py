from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud,ImageColorGenerator
import pandas as pd
import jieba
import re
from imageio.v2 import imread

class Painter:
    def __init__(self):
        pass

    def split_words(self, file_path, encode, index):
        text=''
        df=pd.read_csv(file_path,encoding=encode)
        for comment in df[index]:
            text=text+comment
        # 去除所有非字母、数字、汉字
        text = re.sub(r'[^\w\u4e00-\u9fff]', '', text, flags=re.UNICODE)
        words=jieba.lcut(text)
        return words

    def count_word(self, file_path, encode, index):
        words = self.split_words(file_path, encode, index)
        # 词语长度大于1
        word = [w for w in words if len(w) > 1]
        word_count = dict(Counter(word))
        return word_count

    def cloud_word(self, word_count):
        img=imread('翟天临.jpg')
        wc = WordCloud(
            background_color='white',
            font_path='微软雅黑.ttf',
            max_words=100,
            mask=img,
            max_font_size=500,
            random_state=42,
            width=1550,height=2011,margin=10
        )
        wc.generate_from_frequencies(word_count)
        image_colors=ImageColorGenerator(img)
        plt.imshow(wc.recolor(color_func=image_colors))
        plt.axis('off')
        plt.show()

    def paint(self):
        word_count=self.count_word('comment.csv','utf-8', 'comment')
        self.cloud_word(word_count)

if __name__ == '__main__':
    p = Painter()
    p.paint()