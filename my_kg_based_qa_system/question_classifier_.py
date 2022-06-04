import os


class QuestionClassifier:
    def __init__(self):
        cur_dir = '/'.join(os.path.abspath(__file__).split('/')[:-1])
        # 　特征词路径
        self.person_path = os.path.join(cur_dir, 'person.txt')
        self.movie_path = os.path.join(cur_dir, 'movie.txt')
        self.genre_path = os.path.join(cur_dir, 'genre.txt')

        # 加载特征词
        self.person_wds = [i.strip() for i in open(self.person_path, encoding="utf-8") if i.strip()]  # encoding="utf-8"
        self.movie_wds = [i.strip() for i in open(self.movie_path, encoding="utf-8") if i.strip()]
        self.genre_wds = [i.strip() for i in open(self.genre_path, encoding="utf-8") if i.strip()]

        # 电影专有名词
        self.region_words = set(self.person_wds + self.movie_wds + self.genre_wds)

        # 构建词典
        self.wdtype_dict = self.build_wdtype_dict()
        # 问句疑问词
        # 剧情和演员简介容易冲突
        # 评分
        self.q1_qwds = ['分数', '评分', '现象', '症候', '表现']  # 评分
        # 上映
        self.q2_qwds = ['上映', '首映', '上映时间', '首映时间', '首播', '观看', '上线', '影院', '放映', '时间']
        # 风格
        self.q3_qwds = ['风格', '格调', '类型']
        # 剧情
        self.q4_qwds = ['剧情', '内容', '故事', '简介', '情节', '梗概']
        # 出演
        self.q5_qwds = ['演员', '演的', '出演', '演过', '哪些人']
        # 演员简介
        self.q6_qwds = ['是谁', '介绍', '简介', '谁是', '详细信息', '信息']
        # AB合作
        self.q7_qwds = ['合作', '一起']
        # A一共演过多时
        self.q8_qwds = ['一共', '总共', '多少部', '多少', '参演']
        # A的生日
        self.q9_qwds = ['出生日期', '生日', '出生', '生于']

        print('model init finished ......')

        return

    '''分类主函数'''

    def classify(self, question):
        data = {}
        question_dict = self.check_question(question)
        if not question_dict:
            return {}
        data['args'] = question_dict
        # 收集问句当中所涉及到的实体类型
        types = []
        for type_ in question_dict.values():
            types += type_
        question_type = 'others'

        question_types = []

        # 评分
        if self.check_words(self.q1_qwds, question) and ('movie' in types):
            question_type = 'pingfen'
            question_types.append(question_type)
        # 上映
        if self.check_words(self.q2_qwds, question) and ('movie' in types):
            question_type = 'shangying'
            question_types.append(question_type)

        # 风格
        if self.check_words(self.q3_qwds, question) and ('movie' in types):
            question_type = 'fengge'
            question_types.append(question_type)
        # 剧情
        if self.check_words(self.q4_qwds, question) and ('movie' in types):
            question_type = 'jvqing'
            question_types.append(question_type)
        # 出演
        if self.check_words(self.q5_qwds, question) and ('movie' in types):
            question_type = 'chuyan'
            question_types.append(question_type)

        # 演员简介
        if self.check_words(self.q6_qwds, question) and ('person' in types):
            question_type = 'yanyuanjianjie'
            question_types.append(question_type)
        # 合作出演
        if self.check_words(self.q7_qwds, question) and ('person' in types):
            question_type = 'hezuochuyan'
            question_types.append(question_type)
        # 总共
        if self.check_words(self.q8_qwds, question) and ('person' in types):
            question_type = 'zonggong'
            question_types.append(question_type)

        # 生日
        if self.check_words(self.q9_qwds, question) and ('person' in types):
            question_type = 'shengri'
            question_types.append(question_type)
        # 将多个分类结果进行合并处理，组装成一个字典
        data['question_types'] = question_types

        return data

    '''构造词对应的类型'''

    def build_wdtype_dict(self):
        wd_dict = dict()
        for wd in self.region_words:
            wd_dict[wd] = []
            if wd in self.person_wds:
                wd_dict[wd].append('person')
            if wd in self.movie_wds:
                wd_dict[wd].append('movie')
            if wd in self.genre_wds:
                wd_dict[wd].append('genre')

        return wd_dict

    '''问句过滤'''

    # 实现了把问句中的电影、演员、类型等专有名词提取
    def check_question(self, question):
        final_wds = []
        for wd in self.region_words:
            if wd in question:
                final_wds.append(wd)
        final_dict = {i: self.wdtype_dict.get(i) for i in final_wds}
        return final_dict

    '''基于特征词进行分类'''

    def check_words(self, wds, sent):
        for wd in wds:
            if wd in sent:
                return True
        return False

if __name__ == '__main__':
    handler = QuestionClassifier()
    while 1:
        question = input('input an question:')
        data = handler.classify(question)
        print(data)