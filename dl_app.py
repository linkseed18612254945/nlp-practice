from flask import Flask, jsonify, request
from flask_cors import CORS
import json
import utils
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


@app.route('/classify', methods=['POST'])
def classify():
    post_content = request.get_data().decode()
    post_content = json.loads(post_content)
    data = post_content['data'] if 'data' in post_content else post_content
    contents = [d.get('content') or d.get('text') or d.get('m_content') for d in data]
    tags, scores = text_classify.predict(SystemStatus.model, SystemStatus.tokenizer, contents)
    print(scores)
    print(tags)
    for d, score, label in zip(data, scores, tags):
        d['category'] = ';'.join([SystemStatus.label_map[tag] for tag in label[:SystemStatus.predict_tag_num]])
        d['score'] = ';'.join([str(int(s * 100)) for s in score[:SystemStatus.predict_tag_num]])
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
                and utils.contain_words(content, SystemStatus.yishixingtai_words):
            d['mg_type'] = '意识形态'
            d['to_mq'] = 1
        elif set(d['category'].split(';')).intersection({'社会', '文化艺术', '教育', '民族宗教', '休闲娱乐'}) \
                and utils.contain_words(content, SystemStatus.fengjianmixin_words):
            d['mg_type'] = '封建迷信'
            d['to_mq'] = 1
        elif set(d['category'].split(';')).intersection({'社会', '国内政治', '经济金融'}) \
                and utils.contain_words(content, SystemStatus.changshuaijingji_words):
            d['mg_type'] = '唱衰经济'
            d['to_mq'] = 1
        elif set(d['category'].split(';')).intersection({'社会', '国内政治', '国际政治', '文化艺术'}) \
                and utils.contain_words(content, SystemStatus.lishixuwu_words):
            d['mg_type'] = '历史虚无'
            d['to_mq'] = 1
        else:
            d['to_mq'] = 0
    res = {"data": data} if 'data' in post_content else data
    return jsonify(res)

@app.route('/classify', methods=['POST'])
def classify():
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

def get_n_keywords(text, max_words_num=2):
    words, tags = SystemStatus.preprocessor.cn_sentence_tokenize(text, tag=True)
    use_words = []
    for w, t in zip(words.split(), tags.split()):
        if t[0] == 'n' or t == 'eng':
            use_words.append(w)
        if len(use_words) >= max_words_num:
            break
    return use_words


def search_mpp(keywords):
    keyword = f'"{keywords[0]}"'
    for k in keywords[1:]:
        keyword = keyword + f' AND "{k}"'
    url = "http://10.136.74.88:9211/mpp/select"
    g_asp = 'toutiao.com'
    table_name = 'tp_msg_selfmedia_article_comment'
    start_time, end_time = utils.gap_days(3)
    sql = f"select g_ch_key, m_title, m_board_names, m_category_tag, m_content, category, m_publish_time from {table_name} " \
          f"where unix_timestamp(m_publish_time) between unix_timestamp({start_time}) and unix_timestamp({end_time}) and g_asp={g_asp} " \
          f"and search()='m_content:({keyword})' order by m_publish_time desc limit 100 "
    post_dict = {
        "dbName": "msg_db",
        "sql": sql
    }
    print(sql)
    res = utils.post_dict_return_dict_data(url, post_dict)['data']
    return res


if __name__ == '__main__':
    CORS(app, supports_credentials=True)
    app.run(host="0.0.0.0", port=5050, debug=False)
