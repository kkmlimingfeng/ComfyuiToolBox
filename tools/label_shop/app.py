# -*- coding: utf-8 -*-
"""
标签超市（label_shop）
浏览 / 搜索本地收集的 danbooru 标签与 WD14 v2/v3 selected_tags 数据集：
- 数据取三个来源的并集，每个标签按实际来源显示 danbooru / v2 / v3 徽章
- 两级分类：大类 → 小类（WD14 分类 + 基于中文翻译/英文名的关键词规则推断）
- 搜索：中文名、英文标签、别名，自动忽略下划线 / 横线 / 斜杠 / 空格差异
- 购物车：卡片流式排列，"+" 加入收藏（localStorage 持久化），
  抽屉顶部大文本框实时组合成提示词字符串
数据来源：files/ 下的 danbooru_all.csv（主数据：标签/次数/id/别名/中文/备注）
与 selected_tags v2.csv、v3.csv（WD14 分类与收录标记）。
端口：7868
"""

import csv
import hashlib
import json
import os
import re
import threading
import time

from flask import Flask, jsonify, render_template_string, request

PORT = 7868

HERE = os.path.dirname(os.path.abspath(__file__))
FILES_DIR = os.path.join(HERE, "files")
ALL_CSV = os.path.join(FILES_DIR, "danbooru_all.csv")
V2_CSV = os.path.join(FILES_DIR, "selected_tags v2.csv")
V3_CSV = os.path.join(FILES_DIR, "selected_tags v3.csv")
MERGED_CSV = os.path.join(HERE, "tags_merged.csv")   # build_data.py 产物（含机翻），优先加载
# 可选分类覆盖文件（如 AI 重新规划的分类），只需 name,cat,sub 三列；
# 存在时按标签名覆盖大类/小类，删除或改名即可回到内置分类
CLASSIFIED_CSV = os.path.join(HERE, "tags_merged_classified.csv")

# ---------------- 两级分类规则 ----------------
# (大类, 小类, 中文关键词列表, 英文正则列表)，按顺序匹配，首个命中生效
_R = [
    # 发型
    ("发型", "发色·挑染", ["白发", "黑发", "金发", "棕发", "蓝发", "红发", "绿发", "紫发", "粉发", "发色", "挑染", "渐变色", "彩色头发"],
     [r"(white|black|blond|brown|blue|red|green|purple|pink|orange|grey|gray|silver|multicolored|streaked)_?hair\b"]),
    ("发型", "辫·马尾·髻", ["辫", "马尾", "丸子头", "发髻", "发包", "垂发", "编发"],
     [r"twintails|twin_tails|(twin|side|braided)_(ponytail|braid|bun)|ponytail|\bbraids?\b|hair_bun|drill_hair"]),
    ("发型", "长度·刘海", ["长发", "短发", "中长发", "刘海", "中分", "秃头", "光头", "披头散发"],
     [r"(absurdly_)?(very_)?(long|short|medium)_hair\b|\bbangs?\b|parted_bangs|bald|hime_cut"]),
    ("发型", "样式·质感", ["卷发", "直发", "呆毛", "翘发", "蓬乱", "湿发", "发型", "波浪发", "蓬松", "发梢"],
     [r"wavy_hair|curly_hair|straight_hair|ahoge|messy_hair|wet_hair|spiked_hair|hair_flap|sidelocks|undercut"]),
    # 服装·穿戴
    ("服装·穿戴", "袜·鞋", ["袜", "鞋", "靴", "赤脚", "光脚", "足袋", "凉鞋", "高跟鞋", "拖鞋"],
     [r"\bsocks?\b|thighhighs|pantyhose|kneehighs|\bshoes?\b|\bboots?\b|loafers|sandals|barefoot|pumps|sneakers|footwear"]),
    ("服装·穿戴", "帽·头饰", ["帽子", "头饰", "发饰", "发卡", "头冠", "王冠", "花环", "头巾", "面纱", "耳机", "发带", "护目镜"],
     [r"\bhat\b|\bcap\b|headwear|headband|hair_ornament|hairclip|hair_ribbon|hair_flower|crown|tiara|halo|visor|headphones|goggles|veil"]),
    ("服装·穿戴", "手套·袖", ["手套", "护腕", "袖", "臂环", "袖套"],
     [r"gloves?|fingerless|arm_warmers|wristband|bracer|sleeves|cuff"]),
    ("服装·穿戴", "裙装", ["裙", "连衣裙", "礼服"],
     [r"\bskirt\b|pleated|miniskirt|\bdress\b|gown|sundress"]),
    ("服装·穿戴", "上衣", ["衬衫", "毛衣", "T恤", "背心", "上衣", "衬衣", "吊带"],
     [r"\bshirt\b|dress_shirt|sweater|cardigan|t-shirt|tshirt|\bvest\b|tank_top|blouse|halter|tube_top|hoodie"]),
    ("服装·穿戴", "下装", ["裤", "牛仔", "打底裤"],
     [r"\bpants\b|shorts|jeans|leggings|track_pants|skort"]),
    ("服装·穿戴", "泳装·内衣", ["泳装", "泳衣", "比基尼", "内衣", "胸罩", "丁字裤", "吊袜带"],
     [r"swimsuit|bikini|competition_swimsuit|one-piece|lingerie|\bbras?\b|panties|garter|naked_apron|underwear"]),
    ("服装·穿戴", "外套·制服", ["外套", "夹克", "大衣", "披风", "斗篷", "风衣", "西装", "水手服", "校服", "制服"],
     [r"\bjacket\b|coat\b|cloak|cape\b|poncho|suit|school_uniform|serafuku|uniform|labcoat|overcoat"]),
    ("服装·穿戴", "饰品·配件", ["项链", "戒指", "耳环", "手镯", "项圈", "胸针", "领带", "领结", "围巾", "腰带", "吊坠", "眼镜", "墨镜", "口罩", "面具"],
     [r"necklace|pendant|\brings?\b|earrings|choker|bracelet|brooch|necktie|bowtie|scarf|shawl|\bbelt\b|glasses|sunglasses|eyewear|mask"]),
    ("服装·穿戴", "图案·花色", ["花纹", "图案", "格子", "条纹", "圆点", "迷彩", "蕾丝", "印花", "豹纹", "纯色"],
     [r"striped|plaid|checkered|polka_dot|pinstripe|camouflage|lace|floral_print|leopard|pattern"]),
    ("服装·穿戴", "整体着装", ["服装", "装束", "着装", "盔甲", "和服", "浴衣", "旗袍", "汉服", "巫女服", "女仆装", "修女服"],
     [r"clothes|outfit|armor|kimono|yukata|china_dress|hanfu|miko|maid|nun|bodysuit|catsuit|jumpsuit|robe"]),
    # 眼睛·五官
    ("眼睛·五官", "瞳色", ["瞳色", "异色瞳", "金瞳", "红瞳", "蓝瞳", "碧眼", "黑眼", "紫眼"],
     [r"(blue|brown|green|red|purple|yellow|aqua|pink|orange|grey|gray|black|heterochromia)_(eyes?|pupil)"]),
    ("眼睛·五官", "眼型·眼神", ["眼睛", "眼神", "半闭眼", "闭眼", "睁眼", "眼睫毛", "死鱼眼", "斜视"],
     [r"half-closed_eyes|closed_eyes|open_eyes|\beyes\b|eyelashes|tsurime|tareme|jitome|sideways_glance"]),
    ("眼睛·五官", "眉", ["眉"], [r"\beyebrows?\b"]),
    ("眼睛·五官", "嘴·牙", ["嘴", "唇", "牙", "舌头", "虎牙", "口水"],
     [r"\bmouth\b|lips|open_mouth|closed_mouth|teeth|fangs|tongue|drooling|pout"]),
    ("眼睛·五官", "耳", ["兽耳", "猫耳", "精灵耳", "耳朵", "耳洞"],
     [r"\bears?\b|animal_ears|cat_ears|fox_ears|pointed_ears|bunny_ears"]),
    ("眼睛·五官", "鼻·脸型", ["鼻", "脸颊", "下巴", "雀斑", "泪痣"],
     [r"\bnose\b|cheeks|chin|freckles|round_face"]),
    # 表情·情绪
    ("表情·情绪", "笑", ["微笑", "大笑", "傻笑", "坏笑", "咧嘴", "笑容", "得意笑"],
     [r"smil|laugh|grin|smirk|giggle"]),
    ("表情·情绪", "哭·泪", ["哭", "泪", "抽泣", "嚎啕"],
     [r"cry|crying|tears|sobbing|teary"]),
    ("表情·情绪", "害羞·脸红", ["脸红", "害羞", "羞涩", "难为情"],
     [r"blush|embarrass|flustered"]),
    ("表情·情绪", "愤怒·严肃", ["生气", "愤怒", "皱眉", "严肃", "冷漠", "面无表情", "不悦", "撇嘴"],
     [r"angry|annoyed|frown|scowl|rage|serious|smug|stoic|expressionless"]),
    ("表情·情绪", "惊讶·恐惧", ["惊讶", "吃惊", "震惊", "害怕", "恐惧", "慌张", "紧张", "惊恐", "发抖"],
     [r"surpris|shock|afraid|fear|scared|nervous|panic|trembl|startled"]),
    ("表情·情绪", "其他情绪", ["开心", "高兴", "兴奋", "困", "累", "疲惫", "无聊", "撒娇", "沮丧", "难过", "忧郁", "吐舌", "喘气", "鼓脸", "哭笑"],
     [r"excited|sleepy|tired|exhausted|bored|proud|sulk|sad|depress|melancholy|wink|panting|puffed_cheeks"]),
    # 动作·姿势
    ("动作·姿势", "站·坐·卧", ["站", "坐", "躺", "趴", "蹲", "跪", "盘腿", "翘腿", "悬浮", "漂浮", "睡"],
     [r"standing|sitting|lying|kneeling|crouching|squatting|cross-legged|floating|sleeping|all_fours"]),
    ("动作·姿势", "走跑跳·飞", ["走", "跑", "跳", "飞", "冲刺", "跃"],
     [r"walking|running|jumping|flying|dashing|leaping|mid-air"]),
    ("动作·姿势", "手部动作", ["举起", "抬手", "挥手", "握", "拿", "抱着", "指着", "比心", "鼓掌", "叉腰", "捂", "双手", "伸手", "托"],
     [r"arms_up|raised_arm|waving|holding|grabbing|pointing|clapping|hands_on_hips|covering|reaching|v_hands|outstretched"]),
    ("动作·姿势", "身体动作", ["弯腰", "伸展", "回头", "爬", "倒立", "翻滚", "靠", "支撑", "扭转"],
     [r"bending_over|leaning|stretch|turning_around|climbing|handstand|rolling|against|propping|twisting"]),
    ("动作·姿势", "互动动作", ["拥抱", "牵手", "亲吻", "背", "驮", "骑", "挨着", "贴着", "对视", "互动"],
     [r"hugging|holding_hands|kissing|piggyback|carrying|riding|cuddle|head_to_head|facing|interlocked"]),
    ("动作·姿势", "战斗·运动", ["战斗", "攻击", "挥剑", "拔刀", "射击", "踢", "拳", "投掷", "游泳", "体操"],
     [r"fighting|attack|sword|drawing_sword|shooting|kick|punch|throwing|swimming|gymnastics"]),
    # 身体·体征
    ("身体·体征", "胸部", ["胸", "乳", "巨乳", "贫乳"],
     [r"breasts|cleavage|nipples"]),
    ("身体·体征", "腰·臀", ["腰", "臀", "屁股", "胯"],
     [r"\bwaist\b|hips|butt|glutes"]),
    ("身体·体征", "腿·脚", ["腿", "大腿", "膝盖", "小腿", "脚趾", "趾"],
     [r"legs?|thighs|knees?|calves|\bfeet\b|toes|crossed_legs"]),
    ("身体·体征", "手臂·肩", ["手臂", "胳膊", "肩膀", "腋"],
     [r"arms?|shoulders|armpits"]),
    ("身体·体征", "尾巴·翅膀·角", ["尾巴", "翅膀", "触角", "触手", "鱼鳍", "兽角"],
     [r"tail\b|wings?|\bhorns?\b|tentacles|fins"]),
    ("身体·体征", "皮肤·身材", ["皮肤", "身材", "身高", "丰满", "苗条", "肌肉", "腹肌", "肚子", "痣", "晒痕"],
     [r"skin|\bbody\b|curvy|slim|muscular|\babs\b|stomach|tanlines|birthmark"]),
    ("身体·体征", "年龄·体型", ["萝莉", "御姐", "成熟", "年上", "年下", "巨大", "巨人", "迷你", "体型差"],
     [r"loli|milf|aged|mature|giantess|size_difference|\bmacro\b"]),
    # 场景·环境
    ("场景·环境", "天空·气候", ["天空", "云", "雨", "雪", "风", "雾", "雷", "彩虹", "星空", "月亮", "太阳", "晴天", "阴天"],
     [r"sky\b|clouds?|\brain\b|\bsnow\b|\bwind\b|\bfog\b|thunder|rainbow|starry|\bmoon\b|sun(light)?|sunny|overcast"]),
    ("场景·环境", "自然·植物", ["森林", "树", "花", "草", "山", "田野", "竹林", "草原", "叶子", "花瓣", "花园"],
     [r"forest|trees?|flowers?|grass|mountain|field|bamboo|nature|leaves|petals|garden|jungle"]),
    ("场景·环境", "水边·海洋", ["海", "湖", "河", "溪", "滩", "船", "温泉", "泳池", "水下"],
     [r"ocean|\bsea\b|\bwater\b|lake|river|beach|boat|ship|onsen|hot_spring|\bpool\b|underwater|waterfall"]),
    ("场景·环境", "室内", ["室内", "房间", "教室", "卧室", "厨房", "浴室", "图书馆", "咖啡", "商店", "走廊", "窗", "床上", "客厅", "桌子", "沙发"],
     [r"indoors|\broom\b|classroom|bedroom|kitchen|bathroom|library|cafe|\bshop\b|hallway|window|on_bed|table|sofa"]),
    ("场景·环境", "城市·建筑", ["城市", "街", "建筑", "楼", "桥", "废墟", "神社", "寺庙", "城堡", "塔", "都市", "小巷"],
     [r"city|street|building|bridge|ruins|shrine|temple|castle|tower|urban|alley|skyline"]),
    ("场景·环境", "日夜·季节", ["白天", "夜晚", "黄昏", "黎明", "春天", "夏天", "秋天", "冬天", "季节"],
     [r"\bday\b|night|dusk|sunset|dawn|morning|spring|summer|autumn|winter|season"]),
    # 构图·镜头
    ("构图·镜头", "取景范围", ["特写", "上半身", "全身", "半身", "头像", "胸像", "局部"],
     [r"close-up|upper_body|full_body|portrait|face_focus|headshot|wide_shot|cowboy_shot"]),
    ("构图·镜头", "视角·方向", ["视角", "仰视", "俯视", "正面", "背面", "侧面", "从背后", "从上方", "从下方", "斜角"],
     [r"from_(above|below|behind|side|outside)|view|angle|facing_|looking_at_viewer|\bback\b|profile"]),
    ("构图·镜头", "镜头·景深", ["景深", "虚化", "鱼眼", "广角", "镜头", "微距", "模糊背景"],
     [r"depth_of_field|bokeh|blurry|fisheye|lens|macro|foreshortening"]),
    ("构图·镜头", "构图·留白", ["构图", "留白", "极简", "对称", "边框", "分屏", "剪影"],
     [r"composition|negative_space|minimalist|symmetrical|border|split_screen|silhouette"]),
    # 光影·氛围
    ("光影·氛围", "光源·光线", ["逆光", "阳光", "灯光", "霓虹", "烛光", "发光", "聚光", "丁达尔", "闪烁", "光斑"],
     [r"lighting|backlight|sunlight|lamp|neon|candle|glowing|spotlight|god_rays|lens_flare"]),
    ("光影·氛围", "明暗·影子", ["影", "阴影", "黑暗", "昏暗", "轮廓光"],
     [r"shadow|\bdark\b|\bdim\b|rim_light"]),
    ("光影·氛围", "氛围·情绪感", ["氛围", "梦幻", "诡异", "恐怖", "温馨", "孤独", "浪漫", "神秘", "赛博", "蒸汽波", "复古", "怀旧"],
     [r"atmospheric|dreamy|eerie|horror|cozy|lonely|romantic|mysterious|cyberpunk|vaporwave|retro|nostalgic"]),
    # 物品·道具
    ("物品·道具", "武器·装备", ["剑", "刀", "枪", "弓", "斧", "锤", "盾", "武器", "魔杖", "炸弹", "炮"],
     [r"sword|katana|\bgun\b|rifle|pistol|\bbow\b|axe|hammer|shield|weapon|staff|dagger|bomb|cannon|scythe"]),
    ("物品·道具", "食物·饮品", ["食", "吃", "喝", "咖啡", "茶", "蛋糕", "冰淇淋", "水果", "面", "酒", "饮料", "甜点", "吸管"],
     [r"food|eating|drinking|coffee|\btea\b|cake|ice_cream|fruit|noodles|alcohol|drink|dessert|\bcup\b|straw|lollipop"]),
    ("物品·道具", "电子·数码", ["手机", "电脑", "相机", "电视", "屏幕", "机器人", "游戏机", "机械"],
     [r"phone|computer|laptop|camera|television|screen|robot|game_console|mechanical"]),
    ("物品·道具", "交通·载具", ["车", "摩托", "自行车", "飞机", "列车", "地铁", "载具"],
     [r"\bcar\b|motorcycle|bicycle|airplane|train|\bbus\b|subway|vehicle"]),
    ("物品·道具", "日常物品", ["书", "笔", "伞", "箱子", "包裹", "玩具", "玩偶", "气球", "花束", "行李", "信", "纸", "钱", "钥匙", "铃铛", "镜子", "枕头", "毯子", "椅子", "灯", "桶"],
     [r"\bbook\b|\bpen\b|umbrella|suitcase|\bbag\b|toy|plush|balloon|bouquet|luggage|letter|paper|money|\bkey\b|\bbell\b|mirror|pillow|blanket|chair|lamp|bucket|broom|basket"]),
    # 风格·艺术
    ("风格·艺术", "画风·媒介", ["插画", "漫画", "像素", "水彩", "素描", "水墨", "油画", "厚涂", "手绘", "真人", "写实", "动画风"],
     [r"sketch|watercolor|oil_painting|pixel|ink\b|manga|comic|anime_style|realistic|photo|3d|render|chibi|flat_color"]),
    ("风格·艺术", "色调·颜色", ["色调", "色彩", "黑白", "单色", "灰度", "渐变", "粉彩", "荧光", "深色", "浅色", "白色", "黑色", "红色", "蓝色", "绿色", "黄色", "紫色", "粉色", "金色", "银色"],
     [r"monochrome|greyscale|grayscale|pastel|neon|vivid|muted|sepia|colorful"]),
    ("风格·艺术", "特效·质感", ["特效", "粒子", "烟雾", "火焰", "闪电", "爆炸", "碎片", "气泡", "雪花", "速度线", "集中线", "闪光"],
     [r"particles|smoke|\bfire\b|lightning|explosion|shards|bubbles|snowflakes|speed_lines|focus_lines|sparkle"]),
    # 画质·质量
    ("画质·质量", "高质量", ["高分辨率", "高清", "杰作", "最佳质量", "超高质量", "精细", "官方", "超清", "极致"],
     [r"masterpiece|best_quality|high_quality|highres|absurdres|\bultra\b|official|\b8k\b|\b4k\b|extremely_detailed|intricate"]),
    ("画质·质量", "低质量·负面", ["低质量", "低分辨率", "模糊", "噪点", "压缩", "伪影", "最差", "糟糕", "马赛克"],
     [r"lowres|low_quality|worst|bad_quality|blurry|jpeg_artifacts|compression|mosaic|censored"]),
    # 数量·人数
    ("数量·人数", "人数组合", ["个女孩", "个男孩", "单人", "双人", "多人", "群体", "三人", "全家福", "合影"],
     [r"\b1(girl|boy|other)\b|\bsolo\b|2girls|multiple_(girls|boys|views)|group|crowd|couple|\b3girls|\bduo\b"]),
]

_RULES = [(c, s, kws, [re.compile(p, re.I) for p in rex]) for c, s, kws, rex in _R]


def _compile_megas():
    """把所有规则合并成两条大正则（中文一条 / 英文一条），
    命中的命名组编号即规则序号，比逐条匹配快约 60 倍"""
    zh_parts, en_parts = [], []
    for i, (_cat, _sub, kws, rexes) in enumerate(_R):
        if kws:
            zh_parts.append("(?P<r%d>%s)" % (i, "|".join(map(re.escape, kws))))
        if rexes:
            en_parts.append("(?P<e%d>(?:%s))" % (i, "|".join(rexes)))
    return (
        re.compile("|".join(zh_parts), re.I) if zh_parts else None,
        re.compile("|".join(en_parts), re.I) if en_parts else None,
    )


_ZH_MEGA, _EN_MEGA = _compile_megas()

# 分类缓存：首次全量推断较慢，结果落盘；规则表变化（md5 变）自动失效
_RULES_MD5 = hashlib.md5(json.dumps(_R, ensure_ascii=False).encode("utf-8")).hexdigest()
_CACHE_FILE = os.path.join(HERE, "_cls_cache.json")


def _load_cls_cache():
    if os.path.isfile(_CACHE_FILE):
        try:
            with open(_CACHE_FILE, encoding="utf-8") as f:
                d = json.load(f)
            if d.get("rules") == _RULES_MD5 and isinstance(d.get("cls"), dict):
                return d
        except Exception:
            pass
    return {"rules": _RULES_MD5, "cls": {}}


def _save_cls_cache(cache):
    try:
        with open(_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False)
    except Exception:
        pass

def classify(name, zh):
    """规则推断 (大类, 小类)：中文翻译优先，其次英文标签名。
    用 finditer 收集大正则命中的所有规则，取规则序号最小者，保证规则表顺序优先"""
    best = None
    if zh and _ZH_MEGA is not None:
        for m in _ZH_MEGA.finditer(zh):
            i = int(m.lastgroup[1:])
            if best is None or i < best:
                best = i
    if best is None and _EN_MEGA is not None:
        for m in _EN_MEGA.finditer(name):
            i = int(m.lastgroup[1:])
            if best is None or i < best:
                best = i
    if best is None:
        return None
    cat, sub = _R[best][:2]
    return cat, sub


app = Flask(__name__)


@app.after_request
def no_store(resp):
    # iframe 内嵌时禁止浏览器缓存，保证门户主题参数与代码更新即时生效
    resp.headers["Cache-Control"] = "no-store"
    return resp


def _norm(s):
    """搜索归一化：忽略大小写与 _ - / : 空格，使 long hair / long_hair / longhair 等价"""
    return "".join(s.lower().split()).replace("_", "").replace("-", "").replace("/", "").replace(":", "")


_LOCK = threading.Lock()
_TAGS = None      # [{n, zh, mt, c, cat, sub, a, db, v2, v3, note, sn(list), sz}]
_STATS = None     # {大类: {小类: n}}
_MERGED_MODE = False   # True = 数据来自 tags_merged*.csv，编辑可写回
_CURRENT_CSV = MERGED_CSV   # 当前数据源 CSV（页面可切换；编辑写回当前文件）
_RELOAD = False             # True = 下次 load_tags 强制重新加载（切换数据源用）


def load_tags():
    """优先加载 build_data.py 产出的 tags_merged.csv（含机翻）；
    不存在时从 files/ 三份原始数据现场合并分类（无机翻列）。
    _RELOAD 置位时强制重载（/api/switch_csv 切换数据源用）"""
    global _TAGS, _STATS, _MERGED_MODE, _RELOAD
    with _LOCK:
        if _TAGS is not None and not _RELOAD:
            return
        _RELOAD = False
        _TAGS = None
        t0 = time.time()

        if os.path.isfile(_CURRENT_CSV):
            tags = []
            with open(_CURRENT_CSV, encoding="utf-8-sig", errors="replace") as f:
                for r in csv.DictReader(f):
                    src = r.get("sources") or ""
                    try:
                        cnt = int(r.get("count") or 0)
                    except ValueError:
                        cnt = 0
                    tags.append({
                        "n": (r.get("name") or "").strip(),
                        "zh": (r.get("zh") or "").strip(),
                        "mt": (r.get("zh_mt") or "").strip(),
                        "c": cnt,
                        "cat": (r.get("cat") or "其他").strip(),
                        "sub": (r.get("sub") or "").strip(),
                        "a": [x for x in (r.get("aliases") or "").split("|") if x],
                        "db": "db" in src, "v2": "v2" in src, "v3": "v3" in src,
                        "note": (r.get("note") or "").strip(),
                    })
            for t in tags:
                t["sn"] = [_norm(t["n"])] + [_norm(a) for a in t["a"]]
                t["sz"] = (t["zh"] + " " + t["mt"]).lower()
            # 可选分类覆盖：tags_merged_classified.csv 存在时按标签名覆盖大类/小类
            if os.path.isfile(CLASSIFIED_CSV):
                idx = {t["n"]: t for t in tags}
                n_cover = 0
                with open(CLASSIFIED_CSV, encoding="utf-8-sig", errors="replace") as f:
                    for r in csv.DictReader(f):
                        t = idx.get((r.get("name") or "").strip())
                        if t is None:
                            continue
                        c = (r.get("cat") or "").strip()
                        if c:
                            t["cat"] = c
                            t["sub"] = (r.get("sub") or "").strip()
                            n_cover += 1
                print(f"[load] 分类覆盖: {CLASSIFIED_CSV} 命中 {n_cover}/{len(tags)} 条")
            _TAGS = tags
            stats = {}
            for t in tags:
                stats.setdefault(t["cat"], {}).setdefault(t["sub"] or "", 0)
                stats[t["cat"]][t["sub"] or ""] += 1
            _STATS = stats
            _MERGED_MODE = True   # merged CSV 为数据源，允许编辑写回
            print(f"[load] 数据源: {os.path.basename(_CURRENT_CSV)} 共 {len(tags)} 条，耗时 {time.time()-t0:.1f}s")
            return

        wd = {}  # {tag: {ver: (category, count)}}；WD14 v2 的 name 用空格分隔，统一换成下划线
        for ver, path in (("v2", V2_CSV), ("v3", V3_CSV)):
            with open(path, encoding="utf-8-sig", errors="replace") as f:
                for r in csv.DictReader(f):
                    name = (r.get("name") or "").strip().replace(" ", "_")
                    if name:
                        try:
                            cnt = int((r.get("count") or "0").strip())
                        except ValueError:
                            cnt = 0
                        wd.setdefault(name, {})[ver] = (str(r.get("category", "")).strip(), cnt)

        def new_rec(name, cnt, cat, sub, aliases, zh, note):
            return {"n": name, "zh": zh, "mt": "", "c": cnt, "cat": cat, "sub": sub,
                    "a": aliases, "db": False, "v2": False, "v3": False, "note": note}

        cache = _load_cls_cache()
        cls = cache["cls"]
        dirty = [False]

        def cls_pair(name, zh):
            pair = cls.get(name)
            if pair is None:
                pair = list(classify(name, zh) or ("其他", ""))
                cls[name] = pair
                dirty[0] = True
            return pair[0], pair[1]

        by_name = {}
        with open(ALL_CSV, encoding="utf-8-sig", errors="replace") as f:
            for r in csv.reader(f):
                if not r or not r[0].strip():
                    continue
                name = r[0].strip()
                try:
                    cnt = int(r[1].strip() or 0)
                except ValueError:
                    cnt = 0
                aliases = []
                col3 = r[3].strip() if len(r) > 3 else ""
                zh = r[4].strip() if len(r) > 4 else ""
                note = r[5].strip() if len(r) > 5 else ""

                if col3 == "kontext 指令":
                    rec = new_rec(name, cnt, "指令", "", [], zh, note)
                else:
                    if col3:
                        aliases = [x.strip() for x in col3.split(",") if x.strip()]
                    infos = wd.get(name) or {}
                    if infos:
                        cat0 = next(iter(infos.values()))[0]
                        if cat0 == "9":
                            cat, sub = "分级", ""
                        elif cat0 == "4":
                            cat, sub = "角色", ""
                        else:
                            cat, sub = cls_pair(name, zh)
                    else:
                        cat, sub = cls_pair(name, zh)
                    if not cnt and infos:
                        cnt = max(c for _, c in infos.values())
                    rec = new_rec(name, cnt, cat, sub, aliases, zh, note)
                rec["db"] = True

                old = by_name.get(name)
                if old is None or rec["c"] > old["c"]:
                    by_name[name] = rec

        # v2/v3 独有标签（约 600 个不在 danbooru_all 里）补充进并集
        for name, infos in wd.items():
            if name in by_name:
                for ver in infos:
                    by_name[name][ver] = True
                continue
            cat0 = next(iter(infos.values()))[0]
            if cat0 == "9":
                cat, sub = "分级", ""
            elif cat0 == "4":
                cat, sub = "角色", ""
            else:
                cat, sub = cls_pair(name, "")
            rec = new_rec(name, max(c for _, c in infos.values()), cat, sub, [], "", "")
            for ver in infos:
                rec[ver] = True
            by_name[name] = rec

        if dirty[0]:
            _save_cls_cache(cache)

        tags = list(by_name.values())
        for t in tags:
            t["sn"] = [_norm(t["n"])] + [_norm(a) for a in t["a"]]
            t["sz"] = (t["zh"] + " " + t.get("mt", "")).lower()

        _TAGS = tags
        stats = {}
        for t in tags:
            stats.setdefault(t["cat"], {}).setdefault(t["sub"] or "", 0)
            stats[t["cat"]][t["sub"] or ""] += 1
        _STATS = stats


# ---------------- 页面 ----------------

PAGE = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>标签超市</title>
<style>
  :root { color-scheme: light dark; }
  * { box-sizing: border-box; }
  body.light {
    --bg:#f0f2f5; --card:#ffffff; --border:#e4e6eb; --text:#1c1e21;
    --hover:#f0f2f5; --head:#0b5bb5; --muted:rgba(28,30,33,.55);
    --accent:#0984e3; --zhc:#8a4a00; --mtc:#0b7285; --cartbg:#fff8e6; --cartbd:#f0d9a8;
    color-scheme: light;
  }
  body.dark {
    --bg:#1e1f22; --card:#29292b; --border:#3a3b40; --text:#e6e8ee;
    --hover:#3a3b40; --head:#74b9ff; --muted:rgba(230,232,238,.55);
    --accent:#0984e3; --zhc:#e8b56a; --mtc:#7fd4e8; --cartbg:#2b2716; --cartbd:#5a4c22;
    color-scheme: dark;
  }
  body { margin:0; padding:14px 18px 60px; font-family:"Segoe UI",system-ui,sans-serif;
         background:var(--bg); color:var(--text); }
  .head { display:flex; gap:8px; margin-bottom:8px; position:sticky; top:0; z-index:40;
          background:var(--bg); padding:10px 0 8px; margin-top:-10px; }
  .head #q { flex:1; padding:9px 12px; border-radius:8px; border:1px solid #d0d3d8;
             background:var(--card); color:var(--text); font-size:14px; }
  .head #minc { flex:0 0 auto; width:86px; padding:9px 8px; border-radius:8px; border:1px solid #d0d3d8;
                background:var(--card); color:var(--text); font-size:13px; }
  body.dark .head input { border-color:#3f4147; background:#29292b; }
  .head button { padding:9px 16px; border:none; border-radius:8px; background:var(--accent);
                 color:#fff; font-size:14px; cursor:pointer; white-space:nowrap; }
  .wrap { display:flex; gap:14px; align-items:flex-start; }
  .main { flex:1; min-width:0; }
  .side { width:212px; flex-shrink:0; position:sticky; top:12px; max-height:calc(100vh - 70px);
          overflow-y:auto; background:var(--card); border:1px solid var(--border);
          border-radius:10px; padding:8px; }
  .sitem { display:flex; align-items:center; gap:5px; padding:7px 9px; border-radius:8px;
           cursor:pointer; font-size:13px; user-select:none; }
  .sitem:hover { background:var(--hover); }
  .sitem.on { background:#6c5ce7; color:#fff; }
  .sitem .arr { width:12px; font-size:10px; opacity:.7; flex-shrink:0; }
  .sitem .lbl { flex:1; }
  .sitem .cnt, .ssub .cnt { font-size:10.5px; opacity:.6; flex-shrink:0; }
  .ssubs { margin:2px 0 4px 14px; display:flex; flex-direction:column; gap:1px;
           border-left:2px solid var(--border); padding-left:7px; }
  .ssub { display:flex; justify-content:space-between; align-items:center; gap:6px;
          padding:4px 8px; border-radius:7px; cursor:pointer; font-size:12px;
          color:var(--muted); user-select:none; }
  .ssub:hover { background:var(--hover); }
  .ssub.on { background:#0a9d58; color:#fff; }
  .smeta { margin-top:10px; padding:8px 9px 2px; border-top:1px dashed var(--border);
           font-size:11px; color:var(--muted); line-height:1.8; }
  @media (max-width:760px) {
    .wrap { flex-direction:column; }
    .side { width:auto; position:static; max-height:none; }
  }
  .cnt { font-size:12.5px; color:var(--muted); margin:0 0 8px; }
  .list { columns:210px; column-gap:8px; }
  .tag { position:relative; display:flex; flex-direction:column;
         background:var(--card); border:1px solid var(--border); border-radius:10px;
         padding:7px 30px 7px 10px; margin:0 0 8px; break-inside:avoid; }
  .tag .row1 { display:flex; align-items:flex-start; gap:6px; padding-right:48px; min-width:0; }
  .tag .nm { min-width:0; font-size:13px; font-weight:600; font-family:Consolas,monospace;
             word-break:break-all; line-height:1.35; }
  .tag .zh { font-size:12.5px; color:var(--zhc); margin-top:3px; line-height:1.35;
             word-break:break-all; display:-webkit-box; -webkit-line-clamp:2;
             -webkit-box-orient:vertical; overflow:hidden; }
  .tag .mt { font-size:12px; color:var(--mtc); margin-top:3px; line-height:1.35;
             word-break:break-all; display:-webkit-box; -webkit-line-clamp:2;
             -webkit-box-orient:vertical; overflow:hidden; }
  .tag .mt .mtag { font-size:10px; border:1px solid currentColor; border-radius:6px;
                   padding:0 4px; margin-right:5px; opacity:.75; }
  .tag .meta { display:flex; gap:4px; flex-wrap:wrap; margin-top:5px; align-items:center; }
  .bd { display:inline-block; padding:1px 7px; border-radius:9px; font-size:10.5px; line-height:1.5; }
  .bd.db { background:#0984e3; color:#fff; }
  .bd.v2 { background:#0a9d58; color:#fff; }
  .bd.v3 { background:#e67e22; color:#fff; }
  .bd.cnt { background:var(--hover); color:var(--muted); }
  .plus { position:absolute; top:6px; right:6px; width:22px; height:22px; border-radius:7px;
          border:1px solid var(--border); background:var(--hover); color:var(--text);
          font-size:15px; line-height:1; cursor:pointer; display:flex; align-items:center; justify-content:center; }
  .plus:hover { background:#e67e22; border-color:#e67e22; color:#fff; }
  .plus.in { background:#e67e22; border-color:#e67e22; color:#fff; }
  .edit { position:absolute; top:6px; right:32px; width:22px; height:22px; border-radius:7px;
          border:1px solid var(--border); background:var(--hover); color:var(--muted);
          font-size:12px; line-height:1; cursor:pointer; display:flex; align-items:center; justify-content:center; }
  .edit:hover { background:#0984e3; border-color:#0984e3; color:#fff; }
  /* 编辑标签模态框 */
  .modal { position:fixed; inset:0; z-index:80; display:none; align-items:center; justify-content:center;
           background:rgba(0,0,0,.45); }
  .modal.open { display:flex; }
  .mbox { width:430px; max-width:92vw; max-height:88vh; overflow-y:auto; background:var(--card);
          border:1px solid var(--border); border-radius:12px; padding:16px 18px;
          box-shadow:0 10px 40px rgba(0,0,0,.3); }
  .mbox h3 { margin:0; font-size:15px; word-break:break-all; font-family:Consolas,monospace; }
  .mbox .mcat { font-size:12px; color:var(--muted); margin:4px 0 10px; }
  .mbox label { display:block; font-size:12px; color:var(--muted); margin:10px 0 4px; }
  .mbox input, .mbox textarea { width:100%; box-sizing:border-box; padding:8px 10px; border-radius:8px;
          border:1px solid var(--border); background:var(--hover); color:var(--text); font-size:13px; }
  .mbox textarea { min-height:60px; resize:vertical; font-family:inherit; }
  .mbtns { display:flex; justify-content:flex-end; gap:8px; margin-top:14px; }
  .mbtns button { padding:8px 18px; border-radius:8px; border:none; cursor:pointer; font-size:13px; }
  .mbtns .ok { background:#0984e3; color:#fff; }
  .mbtns .ok:hover { background:#0b6fc0; }
  .mbtns .cancel { background:var(--hover); color:var(--text); }
  /* 加购飞行动画 + 购物车按钮弹跳 */
  .fly { position:fixed; z-index:90; pointer-events:none; border-radius:10px;
         background:var(--card); border:1px solid var(--border); box-shadow:0 4px 18px rgba(0,0,0,.3);
         transition:transform .62s cubic-bezier(.45,-0.05,.6,1), opacity .62s ease; will-change:transform; }
  @keyframes fabBump { 0%{transform:scale(1)} 40%{transform:scale(1.3)} 100%{transform:scale(1)} }
  .cartfab.bump { animation:fabBump .4s ease; }
  .more { display:block; margin:14px auto; padding:9px 22px; border-radius:8px; border:1px solid var(--border);
          background:var(--card); color:var(--text); font-size:13px; cursor:pointer; }
  .empty { text-align:center; opacity:.5; padding:60px 0; font-size:13px; width:100%; }
  .spin { display:inline-block; width:14px; height:14px; border:2px solid var(--accent);
          border-top-color:transparent; border-radius:50%; animation:r .8s linear infinite;
          vertical-align:-2px; margin-right:6px; }
  @keyframes r { to { transform:rotate(360deg); } }
  /* 购物车抽屉 */
  .cartfab { position:fixed; right:22px; bottom:22px; z-index:60; padding:12px 18px; border:none;
             border-radius:28px; background:#e67e22; color:#fff; font-size:14px; cursor:pointer;
             box-shadow:0 4px 14px rgba(0,0,0,.25); }
  .cartfab b { font-size:15px; }
  .drawer { position:fixed; top:0; right:-380px; width:360px; max-width:94vw; height:100vh; z-index:70;
            background:var(--cartbg); border-left:1px solid var(--cartbd); box-shadow:-6px 0 18px rgba(0,0,0,.2);
            transition:right .22s ease; display:flex; flex-direction:column; }
  .drawer.open { right:0; }
  .drawer h3 { margin:0; padding:12px 14px; font-size:15px; border-bottom:1px solid var(--cartbd);
               display:flex; justify-content:space-between; align-items:center; }
  .drawer h3 span { cursor:pointer; font-size:13px; color:var(--muted); }
  .cbody { flex:1; overflow-y:auto; padding:10px 12px; display:flex; flex-direction:column; gap:8px; }
  #cartStr { width:100%; min-height:110px; resize:vertical; border-radius:8px; border:1px solid var(--cartbd);
             padding:8px 10px; font-family:Consolas,monospace; font-size:13px; line-height:1.5;
             background:var(--card); color:var(--text); }
  .chint { font-size:11.5px; color:var(--muted); }
  .ctags { display:flex; flex-wrap:wrap; gap:6px; }
  .ctag { display:inline-flex; align-items:baseline; gap:5px; background:var(--card); border:1px solid var(--border);
          border-radius:8px; padding:4px 8px; font-size:12px; max-width:100%; cursor:grab; }
  .ctag.dragging { opacity:.35; }
  .ctag .n { font-family:Consolas,monospace; font-weight:600; word-break:break-all; }
  .ctag .z { color:var(--zhc); word-break:break-all; }
  .ctag .x { cursor:pointer; color:var(--muted); }
  .ctag .x:hover { color:#e74c3c; }
  .cfoot { padding:10px 12px; border-top:1px solid var(--cartbd); display:flex; gap:8px; }
  .cfoot button { flex:1; padding:9px 0; border:none; border-radius:8px; font-size:13px; cursor:pointer; }
  .cfoot .cp { background:var(--accent); color:#fff; }
  .cfoot .cl { background:var(--hover); color:var(--text); }
  .theme-toggle { position:fixed; left:20px; bottom:20px; font-size:22px; cursor:pointer; z-index:60;
                  background:transparent; border:none; padding:6px 10px; border-radius:6px; }
  body.dark .theme-toggle:hover { background:#38383a; }
  body.light .theme-toggle:hover { background:#e9e9eb; }
  .toast { position:fixed; top:18px; left:50%; transform:translateX(-50%); background:#2ecc71; color:#fff;
           padding:9px 18px; border-radius:8px; font-size:13px; z-index:99; opacity:0; transition:opacity .25s;
           pointer-events:none; }
  .toast.show { opacity:1; }
  /* 全局来源筛选：亮的来源必须包含（交集），可多选，全灭 = 不过滤 */
  .srcs { display:flex; gap:6px; }
  .srcbtn { padding:8px 14px; border-radius:8px; border:1px dashed var(--border);
            background:var(--btn); color:var(--muted); font-size:13px; cursor:pointer;
            white-space:nowrap; user-select:none; opacity:.45; }
  .srcbtn.sdb.on { background:#0984e3; border-color:#0984e3; color:#fff; opacity:1; border-style:solid; }
  .srcbtn.sv2.on { background:#0a9d58; border-color:#0a9d58; color:#fff; opacity:1; border-style:solid; }
  .srcbtn.sv3.on { background:#e67e22; border-color:#e67e22; color:#fff; opacity:1; border-style:solid; }
  .srcnote { font-size:12px; color:var(--muted); margin-left:2px; }
  .csvsel { padding:8px 10px; border-radius:8px; border:1px solid var(--border);
            background:var(--bg); color:var(--text); font-size:13px; max-width:230px; }
</style>
</head>
<body class="light">
  <div class="head">
    <input id="q" placeholder="搜索标签：中文 / 英文 / 别名，带不带下划线都可以（如 long hair、long_hair、长发、/lh）">
    <button onclick="doSearch(true)">搜索</button>
    <input id="minc" type="number" min="0" step="100" placeholder="最低次数" title="只显示使用次数 ≥ 该值的标签，清空恢复全部">
    <span class="srcs">
      <button class="srcbtn sdb on" id="src_db" onclick="toggleSrc('db')" title="要求标签包含 danbooru 来源（交集，可多选；全灭=不过滤）">danbooru</button>
      <button class="srcbtn sv2 on" id="src_v2" onclick="toggleSrc('v2')" title="要求标签包含 WD14 v2 来源（交集，可多选；全灭=不过滤）">v2</button>
      <button class="srcbtn sv3 on" id="src_v3" onclick="toggleSrc('v3')" title="要求标签包含 WD14 v3 来源（交集，可多选；全灭=不过滤）">v3</button>
      <span class="srcnote" id="srcnote" style="display:none">不过滤</span>
    </span>
    <select id="csvsel" class="csvsel" title="切换数据源 CSV；编辑卡片会写回当前选中的文件" onchange="switchCsv(this.value)"></select>
  </div>
  <div id="cnt" class="cnt" style="display:none"></div>
  <div class="wrap">
    <div class="side" id="side"></div>
    <div class="main">
      <div class="list" id="list"><div class="empty">加载中<span class="spin"></span></div></div>
    </div>
  </div>
  <button class="cartfab" id="cartfab" onclick="toggleCart()">🛒 <b id="cartn">0</b></button>
  <div class="drawer" id="drawer">
    <h3>购物车 <span onclick="toggleCart()">收起 ✕</span></h3>
    <div class="cbody">
      <textarea id="cartStr" readonly placeholder="选中的标签会按顺序组合成提示词显示在这里"></textarea>
      <div class="chint" id="chint"></div>
      <div class="ctags" id="ctags"></div>
    </div>
    <div class="cfoot">
      <button class="cp" onclick="copyCart()">复制</button>
      <button class="cl" onclick="clearCart()">清空</button>
    </div>
  </div>
  <div class="toast" id="toast"></div>
  <div class="modal" id="modal" onclick="if(event.target===this) closeEdit()">
    <div class="mbox">
      <h3 id="emName"></h3>
      <div class="mcat" id="emCat"></div>
      <label>中文翻译（人工，最终以它为准显示）</label>
      <input id="emZh" placeholder="填入正确的中文翻译">
      <label>机翻结果（可手动修正）</label>
      <input id="emMt">
      <label>备注</label>
      <textarea id="emNote" placeholder="用法提示、搭配注意等"></textarea>
      <div class="mbtns">
        <button class="cancel" onclick="closeEdit()">取消</button>
        <button class="ok" onclick="saveEdit()">保存</button>
      </div>
    </div>
  </div>
<button class="theme-toggle" id="themeBtn">💡</button>
<script>
// 全局主题协议：URL ?theme= 优先，其次本地记忆；接收门户 postMessage；灯泡点击时反向通知
(function () {
    function applyTheme(t) {
        document.body.className = t;
        try { localStorage.setItem("toolTheme", t); } catch (e) {}
        var btn = document.getElementById("themeBtn");
        if (btn) btn.textContent = t === "dark" ? "💡" : "🌙";
    }
    window.__applyTheme = applyTheme;
    var urlTheme = new URLSearchParams(location.search).get("theme");
    applyTheme(urlTheme || localStorage.getItem("toolTheme") || "light");
    window.addEventListener("message", function (e) {
        if (e.data && e.data.type === "theme") applyTheme(e.data.theme);
    });
    var themeBtn = document.getElementById("themeBtn");
    if (themeBtn) themeBtn.onclick = function () {
        var t = document.body.classList.contains("dark") ? "light" : "dark";
        applyTheme(t);
        try { if (window.parent !== window) window.parent.postMessage({ type: "theme", theme: t }, "*"); } catch (e) {}
    };
})();

const esc = s => String(s ?? '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
let curQ = '', curCat = '全部', curSub = '', curOffset = 0, total = 0, seq = 0, CATDATA = [];
let SRC = { db: 0, v2: 0, v3: 0 }, openCat = '';   // 来源统计 / 侧栏展开的大类
let SRCF = { db: true, v2: true, v3: true };       // 全局来源筛选：亮的来源必须包含（交集），全灭 = 不过滤
function toggleSrc(k) {
    SRCF[k] = !SRCF[k];
    document.getElementById('src_' + k).classList.toggle('on', SRCF[k]);
    // 全灭 = 不过滤，按钮旁给出提示
    const none = !Object.values(SRCF).some(v => v);
    document.getElementById('srcnote').style.display = none ? '' : 'none';
    doSearch(true);
}

// ---------- 购物车 ----------
let cart = {};   // 对象键顺序 = 插入顺序，即组合顺序
try { cart = JSON.parse(localStorage.getItem('label_shop_cart') || '{}'); } catch (e) { cart = {}; }
function saveCart() { try { localStorage.setItem('label_shop_cart', JSON.stringify(cart)); } catch (e) {} }
function cartList() { return Object.keys(cart).map(k => cart[k]); }
function renderCartN() { document.getElementById('cartn').textContent = Object.keys(cart).length; }
function toggleCart() { document.getElementById('drawer').classList.toggle('open'); renderCart(); }
function syncAddBtns() {
    document.querySelectorAll('[data-add]').forEach(b => {
        const on = !!cart[b.dataset.add];
        b.classList.toggle('in', on);
        b.textContent = on ? '✓' : '＋';
    });
}
function addToCart(t, btn) {
    if (cart[t.n]) delete cart[t.n];
    else {
        cart[t.n] = { n: t.n, zh: t.zh };
        if (btn) flyToCart(btn);   // 加入时播放飞行动画
    }
    saveCart(); renderCartN(); renderCart(); syncAddBtns();
}
function rmCart(n) { delete cart[n]; saveCart(); renderCartN(); renderCart(); syncAddBtns(); }
function clearCart() { cart = {}; saveCart(); renderCartN(); renderCart(); syncAddBtns(); }
function renderCart() {
    const items = cartList();
    document.getElementById('cartStr').value = items.map(t => t.n).join(', ');
    document.getElementById('chint').textContent = items.length ? '共 ' + items.length + ' 个标签，拖动可调整组合顺序' : '';
    const box = document.getElementById('ctags');
    box.innerHTML = items.length ? items.map(t =>
        '<span class="ctag" draggable="true" data-n="' + esc(t.n) + '" title="拖动调整顺序"><span class="n">' + esc(t.n) + '</span>'
        + (t.zh ? '<span class="z">' + esc(t.zh) + '</span>' : '')
        + '<span class="x" onclick="rmCart(\'' + esc(t.n).replace(/'/g, "\\'") + '\')">✕</span></span>').join('')
        : '<span class="chint">购物车是空的，点击标签卡片右上角 ＋ 加入</span>';
}
function toast(msg) {
    const t = document.getElementById('toast');
    t.textContent = msg; t.classList.add('show');
    clearTimeout(t._h); t._h = setTimeout(() => t.classList.remove('show'), 1600);
}
function copyCart() {
    const s = document.getElementById('cartStr').value;
    if (!s) { toast('购物车是空的'); return; }
    navigator.clipboard.writeText(s).then(() => toast('已复制 ' + Object.keys(cart).length + ' 个标签')).catch(() => {
        const ta = document.createElement('textarea'); ta.value = s; document.body.appendChild(ta);
        ta.select(); document.execCommand('copy'); ta.remove(); toast('已复制');
    });
}

// ---------- 侧边栏分级菜单 ----------
async function loadCats() {
    const r = await fetch('/api/stats'); const d = await r.json();
    if (!d.ok) return;
    CATDATA = d.cats;
    SRC = { db: d.db, v2: d.v2, v3: d.v3 };
    renderSide();
}
function renderSide() {
    const totalN = CATDATA.reduce((a, c) => a + c[1], 0);
    let html = '<div class="sitem' + (curCat === '全部' ? ' on' : '') + '" onclick="setCat(\'全部\')">'
        + '<span class="arr"></span><span class="lbl">全部</span><span class="cnt">' + totalN.toLocaleString() + '</span></div>';
    html += CATDATA.map(c => {
        const subs = c[2] || [];
        const open = openCat === c[0], sel = curCat === c[0];
        let h = '<div class="sitem' + (sel ? ' on' : '') + '" onclick="itemClick(\'' + c[0] + '\')">'
            + '<span class="arr">' + (subs.length ? (open ? '▾' : '▸') : '') + '</span>'
            + '<span class="lbl">' + esc(c[0]) + '</span><span class="cnt">' + c[1].toLocaleString() + '</span></div>';
        if (open && subs.length) {
            h += '<div class="ssubs"><div class="ssub' + (sel && !curSub ? ' on' : '') + '" onclick="setSubCat(\'' + c[0] + '\',\'\')">全部小类</div>'
                + subs.map(s => '<div class="ssub' + (sel && curSub === s[0] ? ' on' : '')
                    + '" onclick="setSubCat(\'' + c[0] + '\',\'' + s[0] + '\')"><span>' + esc(s[0]) + '</span><span class="cnt">' + s[1].toLocaleString() + '</span></div>').join('')
                + '</div>';
        }
        return h;
    }).join('');
    html += '<div class="smeta">来源统计<br>danbooru ' + SRC.db.toLocaleString()
        + '<br>v2 收录 ' + SRC.v2.toLocaleString() + ' · v3 收录 ' + SRC.v3.toLocaleString() + '</div>';
    document.getElementById('side').innerHTML = html;
}
function itemClick(c) {
    if (curCat === c && openCat === c) { openCat = ''; renderSide(); return; }   // 已选中的再点一次仅收起小类
    curCat = c; curSub = '';
    const cd = CATDATA.find(x => x[0] === c);
    openCat = (cd && cd[2] && cd[2].length) ? c : '';
    doSearch(true); renderSide();
}
function setCat(c) { curCat = c; curSub = ''; if (c === '全部') openCat = ''; doSearch(true); renderSide(); }
function setSubCat(c, s) { curCat = c; curSub = s; openCat = c; doSearch(true); renderSide(); }

// ---------- 搜索 ----------
function fmtC(n) {
    return n >= 1000000 ? (n / 1000000).toFixed(1) + 'M' : n >= 1000 ? Math.round(n / 1000) + 'K' : String(n);
}
function tagHtml(t) {
    const inCart = !!cart[t.n];
    const catText = t.cat + (t.sub ? '·' + t.sub : '');
    TAGCACHE[t.n] = t;
    return '<div class="tag" data-n="' + esc(t.n) + '">'
        + '<div class="row1"><div class="nm">' + esc(t.n) + '</div></div>'
        + (t.zh ? '<div class="zh">' + esc(t.zh) + '</div>' : '')
        + (t.mt ? '<div class="mt"><span class="mtag">机翻</span>' + esc(t.mt) + '</div>' : '')
        + '<div class="meta">'
        + (t.db ? '<span class="bd db" title="出现在 danbooru_all">danbooru</span>' : '')
        + (t.v2 ? '<span class="bd v2" title="WD14 v2 收录">v2</span>' : '')
        + (t.v3 ? '<span class="bd v3" title="WD14 v3 收录">v3</span>' : '')
        + (t.c ? '<span class="bd cnt">' + fmtC(t.c) + '</span>' : '')
        + '<span class="bd cnt">' + esc(catText) + '</span>'
        + '</div>'
        + (t.a && t.a.length ? '<div class="zh" style="opacity:.6;margin-top:2px">别名: ' + t.a.map(esc).join(' / ') + '</div>' : '')
        + '<button class="edit" title="编辑翻译 / 机翻 / 备注" onclick=\'openEdit('
        + JSON.stringify(t.n).replace(/'/g, '&#39;') + ')\'>✎</button>'
        + '<button class="plus' + (inCart ? ' in' : '') + '" data-add="' + esc(t.n) + '" onclick=\'addToCart('
        + JSON.stringify(t).replace(/'/g, '&#39;') + ', this)\' title="' + (inCart ? '从购物车移除' : '加入购物车') + '">' + (inCart ? '✓' : '＋') + '</button></div>';
}
// 编辑保存后按名字找到当前列表里的对应卡片并整体替换
function refreshCard(t) {
    document.querySelectorAll('#list .tag').forEach(el => {
        if (el.dataset.n === t.n) el.outerHTML = tagHtml(t);
    });
}
function flyToCart(btn) {
    const fab = document.getElementById('cartfab');
    const a = btn.getBoundingClientRect(), b = fab.getBoundingClientRect();
    const el = document.createElement('div');
    el.className = 'fly';
    const sx = a.left + a.width / 2, sy = a.top + a.height / 2;
    el.style.left = (sx - a.width / 2) + 'px';
    el.style.top = (sy - a.height / 2) + 'px';
    el.style.width = a.width + 'px';
    el.style.height = a.height + 'px';
    document.body.appendChild(el);
    const tx = b.left + b.width / 2 - sx, ty = b.top + b.height / 2 - sy;
    requestAnimationFrame(() => requestAnimationFrame(() => {
        el.style.transform = 'translate(' + tx + 'px,' + ty + 'px) scale(.15)';
        el.style.opacity = '.25';
    }));
    setTimeout(() => {
        el.remove();
        fab.classList.remove('bump'); void fab.offsetWidth;
        fab.classList.add('bump');
    }, 620);
}
// ---------- 编辑标签 ----------
const TAGCACHE = {};
let EM = '';
function openEdit(name) {
    const t = TAGCACHE[name];
    if (!t) return;
    EM = name;
    document.getElementById('emName').textContent = t.n;
    document.getElementById('emCat').textContent = (t.cat || '') + (t.sub ? '·' + t.sub : '')
        + (t.c ? '  ·  出现 ' + fmtC(t.c) : '');
    document.getElementById('emZh').value = t.zh || '';
    document.getElementById('emMt').value = t.mt || '';
    document.getElementById('emNote').value = t.note || '';
    document.getElementById('modal').classList.add('open');
    setTimeout(() => document.getElementById('emZh').focus(), 60);
}
function closeEdit() {
    document.getElementById('modal').classList.remove('open');
    EM = '';
}
async function saveEdit() {
    if (!EM) return;
    const name = EM;
    const body = {
        name,
        zh: document.getElementById('emZh').value.trim(),
        zh_mt: document.getElementById('emMt').value.trim(),
        note: document.getElementById('emNote').value.trim(),
    };
    const r = await fetch('/api/edit_tag', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
    });
    const d = await r.json();
    if (!d.ok) { toast(d.error || '保存失败'); return; }
    const t = TAGCACHE[name];
    if (t) {
        t.zh = d.zh; t.mt = d.zh_mt; t.note = d.note;
        refreshCard(t);
        if (cart[name]) { cart[name].zh = t.zh; saveCart(); renderCart(); }
        syncAddBtns();
    }
    closeEdit();
    toast('已保存：' + name);
}
let busy = false;   // 加载锁：防止无限滚动并发请求
async function doSearch(reset) {
    if (!reset && busy) return;
    busy = true;
    if (reset) { curOffset = 0; }
    const q = document.getElementById('q').value.trim();
    curQ = q;
    const box = document.getElementById('list');
    const sid = ++seq;
    if (reset) box.innerHTML = '<div class="empty"><span class="spin"></span>搜索中…</div>';
    const r = await fetch('/api/search', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            q, cat: curCat, sub: curSub, offset: curOffset, limit: 200,
            srcs: Object.keys(SRCF).filter(k => SRCF[k]),   // 全灭 = [] = 不过滤
            min_count: +document.getElementById('minc').value || 0   // 最低次数，0 = 不过滤
        })
    });
    const d = await r.json();
    if (sid !== seq) { busy = false; return; }     // 已有更新的请求，丢弃旧结果
    busy = false;
    if (!d.ok) { box.innerHTML = '<div class="empty">' + esc(d.error || '失败') + '</div>'; return; }
    total = d.total;
    const cnt = document.getElementById('cnt');
    cnt.style.display = '';
    cnt.textContent = '搜索到 ' + total.toLocaleString() + ' 个标签';
    const html = d.tags.map(tagHtml).join('');
    if (reset) {
        box.innerHTML = d.tags.length ? html : '<div class="empty">没有匹配的标签</div>';
    } else {
        box.insertAdjacentHTML('beforeend', html);
    }
    curOffset += d.tags.length;
    syncAddBtns();
}
// 无限滚动：接近底部自动追加下一页
window.addEventListener('scroll', () => {
    if (busy || curOffset >= total) return;
    if (window.innerHeight + window.scrollY >= document.documentElement.scrollHeight - 700) doSearch(false);
}, { passive: true });
let deb;
document.getElementById('q').addEventListener('input', () => {
    clearTimeout(deb); deb = setTimeout(() => doSearch(true), 300);
});
document.getElementById('q').addEventListener('keydown', e => { if (e.key === 'Enter') { clearTimeout(deb); doSearch(true); } });
document.getElementById('minc').addEventListener('input', () => {
    clearTimeout(deb); deb = setTimeout(() => doSearch(true), 400);   // 最低次数输入即过滤
});

// ---------- 购物车拖拽排序（事件委托，renderCart 重绘后依然生效）----------
(() => {
    const box = document.getElementById('ctags');
    let dragEl = null;
    box.addEventListener('dragstart', e => {
        dragEl = e.target.closest('.ctag');
        if (!dragEl) return;
        e.dataTransfer.effectAllowed = 'move';
        try { e.dataTransfer.setData('text/plain', dragEl.dataset.n); } catch (err) {}
        dragEl.classList.add('dragging');
    });
    box.addEventListener('dragover', e => {
        if (!dragEl) return;
        e.preventDefault();
        const tgt = e.target.closest('.ctag');
        if (!tgt || tgt === dragEl) return;
        const r = tgt.getBoundingClientRect();
        const before = (e.clientX - r.left) < r.width / 2;
        box.insertBefore(dragEl, before ? tgt : tgt.nextSibling);
    });
    box.addEventListener('drop', e => { if (dragEl) e.preventDefault(); });
    box.addEventListener('dragend', () => {
        if (!dragEl) return;
        dragEl.classList.remove('dragging');
        dragEl = null;
        const nc = {};   // 按 DOM 顺序重建 cart，保持对象键序 = 组合顺序
        box.querySelectorAll('.ctag').forEach(el => { const n = el.dataset.n; if (cart[n]) nc[n] = cart[n]; });
        cart = nc;
        saveCart(); renderCart();
    });
})();
// ---------- 编辑模态框快捷键 ----------
document.addEventListener('keydown', e => {
    if (!document.getElementById('modal').classList.contains('open')) return;
    if (e.key === 'Escape') { e.preventDefault(); closeEdit(); }
    if (e.key === 'Enter' && e.target.id !== 'emNote') { e.preventDefault(); saveEdit(); }
});

renderCartN(); renderCart();

// ---------- 数据源 CSV 切换 ----------
async function loadCsvList() {
    const r = await fetch('/api/csv_list'); const d = await r.json();
    if (!d.ok) return;
    const sel = document.getElementById('csvsel');
    sel.innerHTML = d.files.map(f =>
        '<option value="' + esc(f) + '"' + (f === d.current ? ' selected' : '') + '>' + esc(f) + '</option>').join('');
}
async function switchCsv(f) {
    if (!f) return;
    toast('正在切换数据源…');
    const r = await fetch('/api/switch_csv', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ file: f })
    });
    const d = await r.json();
    if (!d.ok) { toast('切换失败：' + (d.error || '未知错误')); return; }
    if (d.same) { toast('当前已是该数据源'); return; }
    toast('已切换：' + f + '（共 ' + d.total.toLocaleString() + ' 条，编辑将写回该文件）');
    loadCats().then(() => doSearch(true));
}
loadCsvList();
loadCats().then(() => doSearch(true));   // 默认展示热门标签
</script>
</body>
</html>
"""


# ---------------- 接口 ----------------

def save_merged():
    """把内存 _TAGS 写回当前数据源 CSV（列结构与 build_data.py 完全一致）。
    原子写（pid.tmp + os.replace）；调用方不要持有 _LOCK（本函数不加锁，
    CPython 下读列表字段安全，单人工具无并发编辑冲突）。"""
    if not _MERGED_MODE or _TAGS is None:
        return False
    tmp = f"{_CURRENT_CSV}.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "count", "cat", "sub", "zh", "zh_mt", "aliases", "sources", "note"])
        for t in _TAGS:
            src = ",".join([k for k, on in (("db", t["db"]), ("v2", t["v2"]), ("v3", t["v3"])) if on])
            w.writerow([t["n"], t["c"], t["cat"], t["sub"] or "", t["zh"], t["mt"],
                        "|".join(t["a"]), src, t.get("note", "")])
    os.replace(tmp, _CURRENT_CSV)
    return True


@app.route("/api/csv_list")
def api_csv_list():
    """列出可切换的数据源 CSV（工具目录下 tags_merged*.csv，排除分类覆盖文件）"""
    files = sorted(f for f in os.listdir(HERE)
                   if f.startswith("tags_merged") and f.endswith(".csv")
                   and f != "tags_merged_classified.csv")
    return {"ok": True, "files": files, "current": os.path.basename(_CURRENT_CSV)}


@app.route("/api/switch_csv", methods=["POST"])
def api_switch_csv():
    """切换数据源 CSV 并重新加载（编辑写回目标也随切换改变）"""
    global _CURRENT_CSV, _RELOAD
    d = request.get_json(silent=True) or {}
    name = (d.get("file") or "").strip()
    if not name or os.path.basename(name) != name or name == "tags_merged_classified.csv":
        return jsonify({"ok": False, "error": "非法文件名"})
    path = os.path.join(HERE, name)
    if not os.path.isfile(path):
        return jsonify({"ok": False, "error": "文件不存在"})
    if path == _CURRENT_CSV and _TAGS is not None:
        return jsonify({"ok": True, "file": name, "total": len(_TAGS), "same": True})
    _CURRENT_CSV = path
    _RELOAD = True
    load_tags()
    return jsonify({"ok": True, "file": name, "total": len(_TAGS)})


@app.route("/api/edit_tag", methods=["POST"])
def api_edit_tag():
    """人工修正标签信息：中文翻译 / 机翻 / 备注，实时写回 tags_merged.csv"""
    d = request.get_json(silent=True) or {}
    name = (d.get("name") or "").strip()
    if not name:
        return jsonify({"ok": False, "error": "缺少标签名"})
    with _LOCK:
        if _TAGS is None:
            return jsonify({"ok": False, "error": "数据尚未加载完成"})
        t = next((x for x in _TAGS if x["n"] == name), None)
        if t is None:
            return jsonify({"ok": False, "error": "标签不存在"})
        zh = (d.get("zh") or "").strip()
        mt = (d.get("zh_mt") or "").strip()
        note = (d.get("note") or "").strip()
        t["zh"], t["mt"], t["note"] = zh, mt, note
        t["sz"] = (zh + " " + mt).lower()   # 同步搜索索引，修正后的翻译立即可搜
    if not save_merged():
        return jsonify({"ok": False, "error": "当前不是 merged 数据模式，无法写回（可先运行 build_data.py）"})
    return jsonify({"ok": True, "zh": zh, "zh_mt": mt, "note": note})


@app.route("/")
def home():
    return render_template_string(PAGE)


@app.route("/ping")
def ping():
    return {"ok": True, "app": "label-shop"}


@app.route("/api/stats")
def api_stats():
    load_tags()
    # 动态输出全部大类（分类覆盖文件可能引入新类名）："其他"固定放最后，其余按条数降序
    def cat_item(c, subs):
        sub_list = [[s, m] for s, m in sorted(subs.items(), key=lambda kv: -kv[1]) if s]
        return [c, sum(subs.values()), sub_list]

    cats = [cat_item(c, subs) for c, subs in
            sorted(_STATS.items(), key=lambda kv: -sum(kv[1].values()))
            if c != "其他"]
    if "其他" in _STATS:
        cats.append(cat_item("其他", _STATS["其他"]))
    return {
        "ok": True,
        "total": len(_TAGS),
        "csv": os.path.basename(_CURRENT_CSV),
        "cats": cats,
        "db": sum(1 for t in _TAGS if t["db"]),
        "v2": sum(1 for t in _TAGS if t["v2"]),
        "v3": sum(1 for t in _TAGS if t["v3"]),
    }


@app.route("/api/search", methods=["POST"])
def api_search():
    load_tags()
    d = request.get_json(silent=True) or {}
    q = (d.get("q") or "").strip()
    cat = d.get("cat") or "全部"
    sub = d.get("sub") or ""
    try:
        offset = max(0, int(d.get("offset") or 0))
        limit = min(500, max(1, int(d.get("limit") or 200)))
    except ValueError:
        offset, limit = 0, 200

    ql = q.lower()
    # 空格分词多词 AND 匹配：所有词都要命中（例：“手 背后”→ arm_behind_back 手臂藏在背后）
    words = [(w, _norm(w)) for w in ql.split() if w] if ql else []
    srcs = [s for s in (d.get("srcs") or []) if s in ("db", "v2", "v3")]
    try:
        mc = max(0, int(d.get("min_count") or 0))    # 最低次数过滤，0 = 不过滤
    except (TypeError, ValueError):
        mc = 0
    starts, contains = [], []
    for t in _TAGS:
        if cat != "全部" and t["cat"] != cat:
            continue
        if sub and t["sub"] != sub:
            continue
        if srcs and not all(t[s] for s in srcs):   # 来源筛选：交集（亮的来源都必须有），空 = 不过滤
            continue
        if mc and t["c"] < mc:                     # 热度过滤：次数 ≥ 最低次数
            continue
        if not words:
            starts.append(t)
            continue
        # 多词 AND：每个词在英文名 / zh+mt 合并索引 / 别名中任一命中；首词开头命中进前排
        hit = True
        for w, wn in words:
            if wn in t["sn"][0] or w in t["sz"] or any(wn in s for s in t["sn"][1:]):
                continue
            hit = False
            break
        if not hit:
            continue
        w0, w0n = words[0]
        if t["sn"][0].startswith(w0n) or t["sz"].startswith(w0):
            starts.append(t)
        else:
            contains.append(t)
    res = starts + contains
    return {"ok": True, "total": len(res), "tags": res[offset:offset + limit]}


if __name__ == "__main__":
    import webbrowser
    if os.environ.get("TOOL_NO_BROWSER") != "1":  # 被门户拉起时不自动开浏览器
        threading.Timer(1.0, lambda: webbrowser.open(f"http://127.0.0.1:{PORT}")).start()
    threading.Thread(target=load_tags, daemon=True).start()  # 启动即预热数据，页面秒开
    print(f"[label-shop] serving at http://127.0.0.1:{PORT}")
    app.run(host="127.0.0.1", port=PORT, debug=False, threaded=True)
