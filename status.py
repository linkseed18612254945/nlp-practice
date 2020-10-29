import json
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from preprocess import TokenizeProcessor

class SystemStatus:
    """
    表示系统当前状态的类，包含当前使用的分类引擎模型对象、分词预处理对象，以及其他系统当前状态相关参数值。
    新增官方新闻检测模型
    """
    label_map = ['社会', '经济金融', '互联网', '公共安全', '教育', '文化艺术', '国际政治',
                 '法律司法', '国内政治', '科学技术', '行业产业', '环境能源', '医药卫生', '农业农村',
                 '休闲娱乐', '军事', '民族宗教', '体育']
    predict_tag_num = 1
    with open('config.json', 'r', encoding='utf-8') as f:
        config = json.load(f)
    USE_GPU = config.get('USE_GPU')
    GPU_INDEX = config.get('GPU_NUM')
    if USE_GPU and torch.cuda.is_available():
        device = torch.device('cuda:{}'.format(GPU_INDEX))
    else:
        device = torch.device('cpu')
    bert_pretrain_model_path = config.get('BERT_PRETRAIN_MODEL_PATH')
    tokenizer = BertTokenizer.from_pretrained(bert_pretrain_model_path, do_lower_case=True)
    model = BertForSequenceClassification.from_pretrained(bert_pretrain_model_path, num_labels=18,
                                                          output_attentions=False,
                                                          output_hidden_states=False)
    model.to(device)
    save_point = torch.load(config.get('CLASSIFY_MODEL_PATH'), map_location=torch.device('cpu'))
    model.load_state_dict(save_point, strict=False)
    preprocessor = TokenizeProcessor()

    yishixingtai_words = ["左派","右派","自由主义","资本主义","民主普选","私有化","普世价值","意识形态","左派","右派","教员","走资派","稻上飞","无产阶级","工人阶级","河殇","批林","批孔","四旧","自由派","公知","和平演变","小粉红","民族主义","集体记忆","举国体制","中国特色","异化","民主","和平演变","颜色革命","民运","左翼","右翼","文化革命","正确集体记忆","洗脑","因言获罪","良心犯"]
    fengjianmixin_words = ["镇邪","符咒","风水","转运","驱鬼","阴宅","驱邪","挡煞消灾","藏风聚气","大仙","八字","开光"]
    changshuaijingji_words = ["权贵","官僚主义","官商合谋","瞎折腾","农村衰败","老龄化","企业倒闭","员工降薪","通货膨胀","滞胀","失独","空巢","农村老人","环境污染","草原退化","人民币贬值","撂荒","内卷","返乡潮","倒闭潮","跑路潮"]
    lishixuwu_words = ["历史虚无", "虚无主义", "翻案"]
