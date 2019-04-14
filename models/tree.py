import os
from sklearn.tree import DecisionTreeClassifier, export_graphviz


class DecisionTreeClassifier(DecisionTreeClassifier):
    def draw_tree(self, model, feature_names, save_path):
        """
        绘制决策树
        :param model: 决策树模型
        :param feature_names: 结点名称. [list]
        :param save_path: 文件保存路径
        :return:
        """
        # 生成决策树的路径dot文件，保存到save_path
        export_graphviz(model, out_file=save_path,
                        feature_names=feature_names,
                        filled=True, rounded=True,
                        special_characters=True)

        # 替换dot文件中的字体为Microsoft YaHei,以防止中文乱码
        with open(save_path, 'r', encoding='utf-8') as f:
            dot_data = f.read()

        dot_data = dot_data.replace('fontname=helvetica', 'fontname="Microsoft YaHei"')

        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(dot_data)

        # 生成决策树图像，格式默认为png格式
        os.system('dot -Tpng {0} -o {1}'.format(save_path, save_path.replace('dot', 'png')))
