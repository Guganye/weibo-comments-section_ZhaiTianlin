import time
import requests
import csv
import os


class Spider:
    def __init__(self):
        self.header = {
            'user-agent': 'Your user-agent',
            'cookie': 'Your cookie'
        }
        self.request_delay = 1  # 请求延迟时间(秒)

    def writerRow(self, row):
        if not os.path.exists('comment.csv'):
            with open('comment.csv', 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['comment'])

        with open('comment.csv', 'a', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)

    def get_article_id(self, article_url='https://weibo.com/ajax/statuses/mymblog'):
        id_list = []
        try:
            params = {
                'uid': '1343887012',
                'page': '1'
            }
            res = requests.get(article_url, headers=self.header, params=params, timeout=10)
            res.raise_for_status()

            data = res.json()
            if data.get('data', {}).get('list'):
                for article in data['data']['list']:
                    if 'id' in article:
                        id_list.append(article['id'])
            time.sleep(self.request_delay)
        except Exception as e:
            print(f"获取文章ID失败(第1页): {str(e)}")

        return id_list

    def get_comment(self, id_list, comment_url='https://weibo.com/ajax/statuses/buildComments'):
        for weibo_id in id_list:
            try:
                params = {
                    'id': weibo_id,
                    'is_show_bulletin': '2'
                }
                res = requests.get(comment_url, headers=self.header, params=params, timeout=10)
                res.raise_for_status()

                data = res.json()
                if data.get('data'):
                    for comment in data['data']:
                        if 'text_raw' in comment:
                            self.writerRow([comment['text_raw']])

                if 'max_id' in data and data['max_id'] != 0:
                    self.get_tail_comment(data['max_id'], weibo_id, comment_url)

                time.sleep(self.request_delay)
            except Exception as e:
                print(f"获取评论失败(微博ID{weibo_id}): {str(e)}")
                continue

    def get_tail_comment(self, max_id, weibo_id, comment_url):
        while max_id != 0:
            try:
                params = {
                    'id': weibo_id,
                    'is_show_bulletin': '2',
                    'max_id': max_id
                }
                res = requests.get(comment_url, headers=self.header, params=params, timeout=10)
                res.raise_for_status()

                data = res.json()
                if data.get('data'):
                    for comment in data['data']:
                        if 'text_raw' in comment:
                            self.writerRow([comment['text_raw']])

                new_max_id = data.get('max_id', 0)
                if new_max_id == max_id:  # 防止无限循环
                    break
                max_id = new_max_id

                time.sleep(self.request_delay)

            except Exception as e:
                print(f"获取分页评论失败(微博ID{weibo_id}, max_id{max_id}): {str(e)}")
                break

    def spider(self):
        print("开始爬取微博评论...")
        id_list = self.get_article_id()
        if id_list:
            self.get_comment(id_list)
        else:
            print("未获取到任何微博文章ID")
        print("爬取完成")


if __name__ == '__main__':
    s = Spider()
    s.spider()