from flask import Flask, jsonify, request
from flask_cors import CORS
import re
import json
from utils import text_util
import simple_utils
from huggingface import text_classify
from status import SystemStatus


app = Flask(__name__)


@app.route('/hotsearch_predict', methods=['POST'])
def hotsearch_predict():
    """
    对微博数据进行清洗并分词的接口
    """
    post_content = request.get_data().decode()
    post_content = json.loads(post_content)
    data = post_content['data'] if 'data' in post_content else post_content
    contents = [d.get('content') or d.get('text') or d.get('m_content') for d in data]
    labels, scores, _ = text_classify.predict(SystemStatus.model, SystemStatus.tokenizer, contents)
    for d, label in zip(data, labels):
        d['label'] = label
    res = {"data": data} if 'data' in post_content else data
    return jsonify(res)


@app.route('/yzlj/harm_classify', methods=['POST'])
def classify():
    post_content = request.get_data().decode()
    post_content = json.loads(post_content)
    if 'is_harm' in post_content and post_content.get('is_harm') == 1:
        return jsonify({"data": {'category': '无害', 'score': 99}})

    predict_tag_num = SystemStatus.predict_tag_num if ('categoryNum' not in post_content or post_content.get('categoryNum') < 1) else post_content.get('categoryNum')
    data = post_content['data'] if 'data' in post_content else post_content
    contents = data.get('content') or data.get('text') or data.get('m_content')
    tags, scores = text_classify.predict(SystemStatus.model, SystemStatus.tokenizer, contents)
    data['category'] = [SystemStatus.label_map[t] for t in tags[0][:predict_tag_num]]
    data['score'] = [int(s * 100) for s in scores[0][:predict_tag_num]]
    res = {"data": data} if 'data' in post_content else data
    return jsonify(res)


@app.route('/mg_classify', methods=['POST'])
def mg_classify():
    post_content = request.get_data().decode()
    post_content = json.loads(post_content)
    data = post_content['data'] if 'data' in post_content else post_content
    contents = [d.get('content') or d.get('text') or d.get('m_content') for d in data]
    tags, scores = text_classify.predict(SystemStatus.model, SystemStatus.tokenizer, contents)
    for d, content, score, label in zip(data, contents, scores, tags):
        d['category'] = ';'.join([SystemStatus.label_map[tag] for tag in label[:SystemStatus.predict_tag_num]])
        d['score'] = ';'.join([str(int(s * 100)) for s in score[:SystemStatus.predict_tag_num]])
        if set(d['category'].split(';')).intersection({'社会', '国内政治', '国际政治', '文化艺术'}) \
                and simple_utils.contain_words(content, SystemStatus.yishixingtai_words):
            d['mg_type'] = '意识形态'
            d['to_mq'] = 1
        elif set(d['category'].split(';')).intersection({'社会', '文化艺术', '教育', '民族宗教', '休闲娱乐'}) \
                and simple_utils.contain_words(content, SystemStatus.fengjianmixin_words):
            d['mg_type'] = '封建迷信'
            d['to_mq'] = 1
        elif set(d['category'].split(';')).intersection({'社会', '国内政治', '经济金融'}) \
                and simple_utils.contain_words(content, SystemStatus.changshuaijingji_words):
            d['mg_type'] = '唱衰经济'
            d['to_mq'] = 1
        elif set(d['category'].split(';')).intersection({'社会', '国内政治', '国际政治', '文化艺术'}) \
                and simple_utils.contain_words(content, SystemStatus.lishixuwu_words):
            d['mg_type'] = '历史虚无'
            d['to_mq'] = 1
        else:
            d['to_mq'] = 0
    res = {"data": data} if 'data' in post_content else data
    return jsonify(res)

@app.route('/all_classify', methods=['POST'])
def all_classify():
    post_content = request.get_data().decode()
    post_content = json.loads(post_content)
    predict_tag_num = int(post_content['category_num']) if 'category_num' in post_content else SystemStatus.predict_tag_num
    data = post_content['data'] if 'data' in post_content else post_content
    contents = [d.get('content') or d.get('text') or d.get('m_content') for d in data]
    tags, scores = text_classify.predict(SystemStatus.model, SystemStatus.tokenizer, contents)
    for d, content, score, label in zip(data, contents, scores, tags):
        d['category'] = ';'.join([SystemStatus.label_map[tag] for tag in label[:predict_tag_num]])
        d['score'] = ';'.join([str(int(s * 100)) for s in score[:predict_tag_num]])
    res = {"data": data} if 'data' in post_content else data
    return jsonify(res)


@app.route('/fin_classify', methods=['POST'])
def fin_classify():
    post_content = request.get_data().decode()
    post_content = json.loads(post_content)
    predict_tag_num = int(post_content['category_num']) if 'category_num' in post_content else 2
    data = post_content['data'] if 'data' in post_content else post_content
    contents = [d.get('content') or d.get('text') or d.get('m_content') for d in data]

    use_contents = []
    for d in data:
        d['fullname'], d['name'] = entity_check(d['title'], d['content'])
        clean_content = d['content'].lower()
        clean_content = text_util.traditional2simplified(clean_content)
        clean_content = text_util.remove_html_tag(clean_content)
        clean_content = text_util.clean_no_chinese_no_english_no_digital(clean_content)
        use_content = find_use_content(d['title'], clean_content, d['fullname'], d['name'])
        if d['fullname'] is None and d['name'] is None:
            use_content = 'not_hit'
        else:
            use_content = SystemStatus.preprocessor.cn_sentence_tokenize(use_content)
        use_contents.append(use_content)
        d['use_content'] = use_content

    tags, scores = text_classify.predict(SystemStatus.model, SystemStatus.tokenizer, use_contents)
    for d, content, score, label in zip(data, contents, scores, tags):
        if d['use_content'] == 'not_hit':
            d['category'] = ""
            d['emotion'] = ""
            d['score'] = ""
        d['category'] = ';'.join([SystemStatus.label_map[tag] for tag in label[:predict_tag_num]])
        d['emotion'] = SystemStatus.emotion_map[d['category'].split(';')[0]]
        d['score'] = ';'.join([str(int(s * 100)) for s in score[:predict_tag_num]])
    res = {"data": data} if 'data' in post_content else data
    return jsonify(res)

def entity_check(title, content):
    for i, row in SystemStatus.entity_df.iterrows():
        if row['企业名称'] in title:
            return row['企业名称'], row['企业简称']
        if row['企业简称'] in title:
            return row['企业名称'], row['企业简称']
        if row['企业名称'] in content:
            return row['企业名称'], row['企业简称']
        if row['企业简称'] in content:
            return row['企业名称'], row['企业简称']
    return None, None

def find_use_content(title, content, full_name, name):
    content = title + ' ' + content
    use_contents = []
    if full_name is not None and name is not None:
        last_position = 0
        for span in re.finditer('{}|{}'.format(full_name, name), content):
            use_contents.append(content[max(last_position, span.start() - 10): min(len(content), span.end() + 30)])
            last_position = span.end() + 20
    return " ".join(use_contents)


if __name__ == '__main__':
    CORS(app, supports_credentials=True)
    app.run(host="0.0.0.0", port=5051, debug=False)
