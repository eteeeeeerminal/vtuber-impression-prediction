import re
from typing import Sequence
from collections import Counter

import MeCab
import wordcloud
import jaconv

from dataset import DatasetPack, LabeledDataset

# 第一オノマトペだけ
# 第二オノマトペも含めて
# その他の印象をやる。形態素解析で単語分割するかな

word_to_romaji = {
    'ふわふわ': 'fuwafuwa', 'きらきら': 'kirakira', 'はきはき': 'hakihaki', 'ほわほわ': 'howahowa', 'さらさら': 'sarasara', 'こってり': 'kotteri', 'おっとり': 'ottori', 'わいわい': 'waiwai', 'しっとり': 'sittori',
    'きゃぴきゃぴ': 'kyapikyapi', 'だらだら': 'daradara', 'ゆるゆる': 'yuruyuru', 'るんるん': 'runrun', 'ぎらぎら': 'giragira', 'しとしと': 'shitoshito', 'わちゃわちゃ': 'watyawatya', 'わくわく': 'wakuwaku', 'てきぱき': 'tekipaki',
    'ぴこぴこ': 'picopico', 'ぐだぐだ': 'gudaguda', 'もふもふ': 'mofumofu', 'ぼそぼそ': 'bosoboso', 'ほんわか': 'honwaka', 'しっかり': 'sikkari', 'わんわん': 'wanwan', 'ゆったり': 'yuttari', 'まったり': 'mattari', 'ぽわぽわ': 'powapowa',
    'さばさば': 'sabasaba', 'どんどん': 'dondon', 'へなへな': 'henahena', 'かぷかぷ': 'kapukapu', 'ゆらゆら': 'yurayura', 'ころころ': 'korokoro', 'にこにこ': 'niconico', 'かくかく': 'kakukaku', 'つんつん': 'tsuntsun',
    'あっさり': 'assari', 'のりのり': 'norinori', 'のほほん': 'nohohon', 'しずしず': 'sizusizu', 'たんたん': 'tantan', 'めらめら': 'meramera', 'きりきり': 'kirikiri', 'しんしん': 'shinshin', 'さわさわ': 'sawasawa',
    'ぷかぷか': 'pucapuca', 'ぺらぺら': 'perapera', 'ねむねむ': 'nemunemu', 'きりっ': 'kirixtsu', 'がやがや': 'gayagaya', 'よちよち': 'yochiyochi', 'ふんわり': 'funwari', 'ゆるふわ': 'yurufuwa', 'ねっとり': 'nettori',
    'とんとん': 'tonton', 'ぴかぴか': 'picapica', 'かちかち': 'kachikachi', 'めかめか': 'mekameka', 'ざらざら': 'zarazara', 'にゃんにゃん': 'nyannyan', 'くねくね': 'kunekune', 'もこもこ': 'mokomoko',
    'ほわきゅあ': 'howakyua', 'きゃらきゃら': 'kyarakyara', 'こんこん': 'konkon', 'すたすた': 'sutasuta', 'どんより': 'donyori', 'おどおど': 'odoodo', 'がおー': 'gao', 'うぃーん': 'uin', 'きらっ': 'kiraxtsu', 'すくすく': 'sukusuku',
    'めろめろ': 'meromero', 'はわわ': 'hawawa', 'びりびり': 'biribiri', 'うっふん': 'uhhun', 'かたかた': 'katakata', 'あおあお': 'aoao', 'しゅばっ': 'shubaxtsu', 'わやわや': 'wayawaya', 'がおがお': 'gaogao',
    'ことこと': 'kotokoto', 'ぴよぴよ': 'piyopiyo', 'うきうき': 'ukiuki', 'たどたど': 'tadotado', 'さくさく': 'sakusaku', 'きっちり': 'kitchiri', 'しゃきしゃき': 'shakishaki', 'さっぱり': 'sappari', 'のろのろ': 'noronoro',
    'のんびり': 'nonbiri', 'ひんやり': 'hinyari', 'とげとげ': 'togetoge', 'ぞわぞわ': 'zowazowa', 'ゆりゆり': 'yuriyuri', 'ばりばり': 'baribari', 'ぽかぽか': 'pokapoka', 'ぽこぽこ': 'pokopoko', 'すらすら': 'surasura',
    'いけいけ': 'ikeike', 'ほのぼの': 'honobono', 'じめじめ': 'zimezime', 'いきいき': 'ikiiki', 'ぐちゃぐちゃ': 'gutyagutya', 'がつがつ': 'gatsugatsu', 'かみかみ': 'kamikami', 'ふぇあふぇあ': 'fueafuea', 'くりくり': 'kurikuri',
    'おちおち': 'ochiochi', 'もりもり': 'morimori', 'ふわひかっ': 'fuwahicaxtsu', 'かっぺり': 'kapperi', 'すんっ': 'sunxtsu', 'さばっり': 'sabarri', 'ちぐばぐ': 'chiguhagu', 'もじっこ': 'mozikko', 'だらゆる': 'darayuru',
    'ふわさば': 'fuwasaba', 'もんふり': 'monfuri', 'またーり': 'mata-ri', 'さっすー': 'sassu-', 'さわしと': 'sawashito', 'ぐらりどし': 'guraridoshi', 'へろぐにゃ': 'herogunya', 'さらぬら': 'saranura', 'きらしゅた': 'kirasyuta',
    'さらじわ': 'saraziwa', 'ゆるぴか': 'yurupica', 'ぱっさり': 'passari', 'さーはー': 'sa-ha-', 'ふらほわ': 'furahowa', 'ちゃっかり': 'chakkari', 'からから': 'karakara', 'きらぴか': 'kirapica', 'あっはー': 'ahha-',
    'きびきゃら': 'kibikyara', 'さらしと': 'sarashito', 'ぞくぞく': 'zokuzoku', 'にたー': 'nita-', 'ぴょこぴょこ': 'pyokopyoko', 'けらけら': 'kerakera', 'はつらつ': 'hatsuratsu', 'ひょっこり': 'hyokkori', 'どしどし': 'doshidoshi',
    'ぽくぽく': 'pokupoku', 'びしばし': 'bishibashi', 'べらべら': 'berabera', 'じゃきんじゃきん': 'jakinjakin', 'つよつよ': 'tsuyotsuyo', 'りんごん': 'ringon', 'ぬくぬく': 'nukunuku', 'えへへ': 'ehehe', 'ぎゃはは': 'gyahaha',
    'じとーっ': 'zito-', 'ごごごご': 'gogogogo', 'にょきにょき': 'nyokinyoki', 'もちもち': 'mochimochi', 'べべん': 'beben', 'きんこんかんこん': 'kinkonkankon', 'よしよし': 'yoshiyoshi', 'ぎょろぎょろ': 'gyorogyoro', 'かきかき': 'kakikaki',
    'ふりふり': 'furifuri', 'すやすや': 'suyasuya', 'ひゅーどろどろ': 'hyu-dorodoro', 'ぞっと': 'zotto', 'りんりん': 'rinrin', 'ぎくっ': 'gikuxtsu', 'ひょこっ': 'hyokoxtsu', 'ふらふら': 'furafura', 'ちゅーちゅー': 'chu-chu-', 'すん': 'sun',
    'ごちゃごちゃ': 'gochagocha', 'いっひっひ': 'ihhihhi', 'よぼよぼ': 'yoboyobo', 'すらっ': 'suraxtsu', 'ちゃらちゃら': 'charachara', 'すー': 'su-', 'かっちり': 'kattiri', 'すっきり': 'sukkiri', 'もやもや': 'moyamoya', 'へのへの': 'henoheno',
    'くんくん': 'kunkun', 'ぽそぽそ': 'posoposo', 'がそがそ': 'gasogaso', 'ごのごの': 'gonogono', 'ねろねろ': 'neronero', 'ごそごそ': 'gosogoso', 'すとーん': 'suto-n', 'ばさばさ': 'basabasa', 'ふなふな': 'funafuna',
    'ちんぷんかんぷん': 'chinpunkanpun', 'きゅるるん': 'kyururun', 'ほわん': 'howan', 'きゃぴだら': 'kyapidara', 'じんわり': 'zinwari', 'がばがば': 'gabagaba', 'がつん': 'gatsun', 'はぴはぴ': 'hapihapi', 'ほんわり': 'honwari',
    'ひそひそ': 'hisohiso', 'ゆるり': 'yururi', 'きらきゃぴ': 'kirakyapi', 'ぽわんぽわん': 'powanpowan', 'そよそよ': 'soyosoyo', 'きゅるきゅる': 'kyurukyuru', 'しんと': 'sinto', 'てくてく': 'tekuteku', 'てちてち': 'techitechi',
    'ほわー': 'howa-', 'こっこっ': 'kokkoxtsu', 'ふわー': 'fuwa-', 'ぴょんぴょん': 'pyonpyon', 'にへら': 'nihera', 'さすさす': 'sasusasu', 'はっきり': 'hakkiri', 'のぺー': 'nope-', 'こつこつ': 'kotukotu', 'だーっ': 'da-xtsu',
    'ぽちゃあ': 'pochaa', 'だるだる': 'darudaru', 'がめがめ': 'gamegame', 'ぽえぽえ': 'poepoe', 'くすくす': 'kusukusu', 'きゃわきゃわ': 'kyawakyawa', 'めぇめぇ': 'meemee', 'ちかちか': 'chikachika', 'なみなみ': 'naminami',
    'せかせか': 'sekaseka', 'どよん': 'doyon', 'しこしこ': 'shikoshiko', 'ざわざわ': 'zawazawa', 'がたがた': 'gatagata', 'がしがし': 'gashigashi', 'はらはら': 'harahara', 'ひらひら': 'hirahira', 'おらおら': 'oraora',
    'ゆっさり': 'yussari', 'ちぐはぐ': 'chiguhagu', 'さらすわ': 'sarasuwa', 'わきゃふわ': 'wakyafuwa', 'わさしゃわ': 'wasasyawa', 'わかぽこ': 'wakapoko', 'つんふわ': 'tunfuwa', 'ゆったりほこほこ': 'yuttarihokohoko', 'さわゆら': 'sawayura',
    'ふんわりもこもこ': 'funwarimokomoko', 'がちゃこちゃ': 'gachakocha', 'わちゃすちゃ': 'wachasucha', 'ふんわか': 'funwaka', 'ぎらきり': 'girakiri', 'さらさわもや': 'sarasawamoya', 'わくほか': 'wakuhoka', 'わちゃくちゃ': 'wachakucha',
    'うっとり': 'uttori', 'もくもく': 'mokumoku', 'ねちょねちょ': 'nechonecho', 'ぺたぺた': 'petapeta', 'ほっこり': 'hokkori', 'しゃっきり': 'shakkiri', 'べたべた': 'betabeta', 'ぎゃあぎゃあ': 'gyaagyaa', 'ふさふさ': 'fusafusa', 'じゃんじゃん': 'janjan',
    'きびきび': 'kibikibi', 'ひょろひょろ': 'hyorohyoro', 'へらへら': 'herahera', 'よろよろ': 'yoroyoro', 'ほろほろ': 'horohoro', 'すぱすぱ': 'supasupa', 'どろどろ': 'dorodoro', 'ぱくぱく': 'pakupaku', 'こてこて': 'kotekote',
    'けろけろ': 'kerokero', 'ごろごろ': 'gorogoro', 'きゃっきゃっ': 'kyakkyaxtsu', 'のびのび': 'nobinobi', 'ぴたぴた': 'pitapita', 'つやつや': 'tsuyatsuya', 'ふにゃふにゃ': 'funyafunya', 'てれてれ': 'teretere', 'いちゃいちゃ': 'ichaicha', 'ぎゃーぎゃー': 'gya-gya-',
    'へにょへにょ': 'henyohenyo', 'ばぶばぶ': 'babubabu', 'はむはむ': 'hamuhamu', 'ふよふよ': 'fuyofuyo', 'きちきち': 'kichikichi', 'ばーん': 'ba-n', 'そわそわ': 'sowasowa', 'こそこそ': 'kosokoso', 'ちくちく': 'chikuchiku',
    'うっかり': 'ukkari', 'ろりろり': 'rorirori', 'ぱよぱよ': 'payopayo', 'さーさー': 'sa-sa-', 'さめさめ': 'samesame', 'ばびばび': 'babibabi', 'ゆるーり': 'yuru-ri', 'くまくま': 'kumakuma', 'だんだん': 'dandan', 'あむあむ': 'amuamu',
    'くちくち': 'kuchikuchi', 'ぽわーん': 'powa-n', 'きゃるーん': 'kyaru-n', 'ちゃいちゃい': 'chaichai', 'にくにく': 'nikuniku', 'きむきむ': 'kimukimu', 'みどほの': 'midohono', 'ぺんぺん': 'penpen', 'うにうに': 'uniuni',
    'しるしる': 'shirushiru', 'もふかわ': 'mofukawa', 'わかわか': 'wakawaka', 'ごすごす': 'gosugosu', 'ましゅましゅ': 'masyumasyu', 'ろぼろぼ': 'roborobo', 'ひゅーひゅー': 'hyu-hyu-', 'くろぐろ': 'kuroguro', 'ねこねこ': 'nekoneko',
    'さあさあ': 'saah', 'こーん': 'ko-n', 'ぶんぶん': 'bunbun', 'きらーん': 'kira-n', 'みみみ': 'mimimi', 'いろいろ': 'iroiro', 'はすはす': 'hasuhasu', 'かわかわ': 'kawakawa', 'えるえる': 'erueru', 'ずんずん': 'zunzun', 'おしおし': 'oshioshi',
    'さらり': 'sarari', 'ずがーん': 'zuga-n', 'さわやか': 'sawayaka', 'しょたーん': 'shota-n', 'さるさる': 'sarusaru', 'がさごそ': 'gasagoso', 'しろしろ': 'sirosiro', 'でしでし': 'deshideshi', 'うんぬ': 'unnnu', 'ざざーん': 'zaza-n',
    'きざきざ': 'gizagiza', 'しんみり': 'sinmiri', 'とろん': 'toron', 'しょたしょた': 'shotashota', 'あめあめ': 'ameame', 'のっぺり': 'nopperi', 'やさやさ': 'yasayasa', 'きゃー': 'kya-', 'ひやひや': 'hiyahiya'
}

def normalize_word(word: str, convert_romjaji: bool = True) -> str:
    word = word.strip()
    word = jaconv.kata2hira(word)
    if convert_romjaji:
        word = word_to_romaji.get(word, '')
    return word

class WordCloud:
    FPATH = "./font/ipaexg.ttf"
    def __init__(self) -> None:
        self.tagger = MeCab.Tagger()

    def tokenize(self, sentence: str) -> list[str]:
        words: list[str] = self.tagger.parse(sentence).split("\n")
        ret_words = []
        for w in words:
            if "\t" not in w:
                # ゴミはスキップ
                continue

            word, info = w.split("\t")
            pos = info.split(",")[0]
            if pos in ["助詞", "助動詞", "補助記号"]:
                continue
            ret_words.append(word)

        return ret_words

    def generate(self, words: Sequence[str], save_path: str):
        words = list(map(normalize_word, words))
        counter = Counter(words)
        print(counter)
        print(f"オノマトペの種類：{len(counter.keys())}")

        self.wordcloud = wordcloud.WordCloud(
            background_color="white", font_path=self.FPATH,
            random_state=111,
            width=600, height=400, min_font_size=12
        )
        self.wordcloud.generate(" ".join(words))
        self.wordcloud.to_file(save_path)

    def generate_first_onom_cloud(self, dataset: DatasetPack, save_path: str):
        onoms = map(lambda x: x.origin.label.first_onom, dataset.dataset)
        self.generate(onoms, save_path)

    def generate_all_onom_cloud(self, dataset: DatasetPack, save_path: str):
        first_onoms = list(map(lambda x: x.origin.label.first_onom, dataset.dataset))
        other_onoms = map(lambda x: re.split("[，、 　]", x.origin.label.other_onom), dataset.dataset)
        other_onoms = sum(other_onoms, [])
        other_onoms = list(filter(lambda x: x, other_onoms))
        self.generate(first_onoms + other_onoms, save_path)

    def generate_impression_cloud(self, dataset: DatasetPack, save_path: str):
        impressions = map(lambda x: x.origin.label.other_impressions, dataset.dataset)
        words = map(lambda x: self.tokenize(x), impressions)
        words = sum(words, [])
        self.generate(words, save_path)
