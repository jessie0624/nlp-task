import logging
from enum import Enum
from typing import List, Optional, Mapping

from pydantic import BaseModel, validator

logger = logging.getLogger(__name__)

class TagSpan(BaseModel):
    label: str
    start: int
    end: int

    @validator("label")
    def check_label(cls, label: str):
        if label == "":
            raise ValueError("label must be non empty")
        return label
    
    @validator("start")
    def check_start(cls, start: int):
        if start < 0:
            raise ValueError("start must >= 0")
        return start
    
    @validator("end")
    def check_end(cls, end: int, values):
        if "start" in values and end <= values["start"]:
            raise ValueError("end must > start")
        return end
    
    def __hash__(self):
        return hash((type(self),) + tuple(self.__dict__.values()))

class Entity(TagSpan):
    text: str

    @validator("text")
    def check_text(cls, text: str):
        if text == '':
            raise ValueError("text must be non empty")
        return text
    
    def __hash__(self):
        return hash((type(self), ) + tuple(self.__dict__.values()))

class EntScheme(str, Enum):
    IOB1 = "IOB1"
    IOB2 = "IOB2"
    BIOES = 'BIOES'
    BMES = 'BMES'
    BMESO = 'BMESO'

class EntLabel(str, Enum):
    """
    refer Spacy
    类别	描述
    PERSON	人，包括虚构
    NORP	民族、宗教或政治团体
    FAC	建筑物、机场、公路、桥梁等
    ORG	公司、中介、事业单位等
    GPE	国家、城市、州
    LOC	地理地点、山脉、水体
    PRODUCT	物品、车辆、食物等(不包括服务)
    EVENT	以飓风、战役、战争、体育赛事等命名
    WORK_OF_ART	书籍、歌曲等
    LAW	制成法律的文件
    LANGUAGE	任意语言
    DATE	绝对或相对的日期或时期
    TIME	比一天短的时间
    PERCENT	百分比，包含%
    MONEY	货币价值，包括单位
    QUANTITY	度量，如重量或距离
    ORDINAL	序数词，如第一第二
    CARDINAL	数量词
    """
    PERSON = 'PER' # People.
    NORP = 'NORP' # Nationalities or religious or political groups.
    FAC = 'FAC' # Buildings, airports, highways, bridges, etc.
    ORG = 'ORG' # Companies, agencies, institutions, ect
    GPE = 'GPE' # Countries, cities, state.
    LOC = 'LOC' 
    PRODUCT = 'PRODUCT'
    EVENT = 'EVENT'
    WORK_OF_ART = 'WORK_OF_ART'
    LAW = 'LAW'
    LANGUAGE = 'LANGUAGE'
    DATE = 'DATE'
    TIME = 'TIME'
    PERCENT = 'PERCENT'
    MONEY = 'MONEY'
    QUANTITY = 'QUANTITY'
    ORDINAL = 'ORDINAL'
    CARDINAL = 'CARDINAL'
    PROP = 'PROR' # 其他专有名词

    @classmethod
    def mapping(cls, label: str) -> Optional["EntLabel"]:
        return aliases.get(label, None)

aliases = {'PERSON': EntLabel.PERSON,
           'ORG': EntLabel.ORG,
           'LOC': EntLabel.LOC,
           'GPE': EntLabel.GPE}

class EntTag(BaseModel):
    scheme: EntScheme
    tag: str
    score: float = 1.0

def ent_tags_to_spans(scheme: EntScheme):
    if scheme == EntScheme.IOB2:
        return iob2_to_spans
    elif scheme == EntScheme.BMES:
        return bmes_to_spans
    elif scheme == EntScheme.BMESO:
        return bmeso_to_spans
    elif scheme == EntScheme.BIOES:
        return bioes_to_spans
    else:
        raise ValueError(f"not support {scheme}")

################
# tags to spans
################
def iob2_to_spans(tags: List[str], ignore_labels=None) -> List[TagSpan]:
    ignore_labels = set(ignore_labels) if ignore_labels else set()

    spans = []
    prev_bio_tag = None
    for idx, tag in enumerate(tags):
        tag = tag.upper()
        bio_tag, label = tag[:1], tag[2:]
        if bio_tag == "B":
            spans.append((label, [idx, idx]))
        elif bio_tag == "I" and prev_bio_tag in ("B", "I") and label == spans[-1][0]:
            spans[-1][1][1] = idx
        elif bio_tag == "O":  # o tag does not count
            pass
        else:
            spans.append((label, [idx, idx]))
        prev_bio_tag = bio_tag
    return [
        TagSpan(label=span[0], start=span[1][0], end=span[1][1] + 1) for span in spans if span[0] not in ignore_labels
    ]

def bmes_to_spans(tags: List[str], ignore_labels=None) -> List[TagSpan]:
    ignore_labels = set(ignore_labels) if ignore_labels else set()

    spans = []
    prev_bmes_tag = None
    for idx, tag in enumerate(tags):
        tag = tag.upper()
        bmes_tag, label = tag[:1], tag[2:]
        if bmes_tag in ("B", "S"):
            spans.append((label, [idx, idx]))
        elif bmes_tag in ("M", "E") and prev_bmes_tag in ("B", "M") and label == spans[-1][0]:
            spans[-1][1][1] = idx
        else:
            spans.append((label, [idx, idx]))
        prev_bmes_tag = bmes_tag
    return [
        TagSpan(label=span[0], start=span[1][0], end=span[1][1] + 1) for span in spans if span[0] not in ignore_labels
    ]

def bmeso_to_spans(tags: List[str], ignore_labels=None) -> List[TagSpan]:
    ignore_labels = set(ignore_labels) if ignore_labels else set()

    spans = []
    prev_bmes_tag = None
    for idx, tag in enumerate(tags):
        tag = tag.upper()
        bmes_tag, label = tag[:1], tag[2:]
        if bmes_tag in ("B", "S"):
            spans.append((label, [idx, idx]))
        elif bmes_tag in ("M", "E") and prev_bmes_tag in ("B", "M") and label == spans[-1][0]:
            spans[-1][1][1] = idx
        elif bmes_tag == "O":
            pass
        else:
            spans.append((label, [idx, idx]))
        prev_bmes_tag = bmes_tag
    output = [
        TagSpan(label=span[0], start=span[1][0], end=span[1][1] + 1) for span in spans if span[0] not in ignore_labels
    ] 
    return output   

def bioes_to_spans(tags: List[str], ignore_labels=None) -> List[TagSpan]:
    ignore_labels = set(ignore_labels) if ignore_labels else set()

    spans = []
    prev_bioes_tag = None
    for idx, tag in enumerate(tags): # tags:BPERSON
        tag = tag.upper()
        bioes_tag, label = tag[:1], tag[2:]
        if bioes_tag in ("B", "S"):
            spans.append((label, [idx, idx])) 
        elif bioes_tag in ("I", "E") and prev_bioes_tag in ("B", "I") and label == spans[-1][0]:
            spans[-1][1][1] = idx
        elif bioes_tag == "O":
            pass
        else:
            spans.append((label, [idx, idx]))
        prev_bioes_tag = bioes_tag
    return [
        TagSpan(label=span[0], start=span[1][0], end=span[1][1]+1) for span in spans if span[0] not in ignore_labels
    ]


#################
# spans to tags
#################
def spans_to_iob2(spans: List[TagSpan], text_len: int) -> List[str]:
    sorted_spans = sorted(spans, key=lambda x: x.start)
    tags = []
    offset = 0
    for span in sorted_spans:
        start = span.start
        end = span.end
        for _ in range(offset, start):
            tags.append("O")
        tags.append(f"B-{span.label}")
        for _ in range(start+1, end):
            tags.append(f"I-{span.label}")
        offset = end
    
    for _ in range(offset, text_len):
        tags.append("O")
    assert len(tags) == text_len, f"spans: {spans}"
    return tags

def spans_to_entities(spans: List[TagSpan], text:str) -> List[Entity]:
    entities = []
    for span in spans:
        value = text[span.start : span.end]
        entities.append(Entity(label=span.label, start=span.start, end=span.end, text=value))
    return entities


#################
# tags to tags
#################
def iob1_to_iob2(tags: List[str]) -> List[str]:
    """
    检查数据是否是合法的IOB数据，如果是IOB1会被自动转换为IOB2,两者的区别见
    https://datascience.stackexchange.com/questions/37824/difference-between-iob-and-iob2-format
    """
    output = []
    for i, tag in enumerate(tags):
        tag = tag.upper()
        if tag == "O":
            output.append(tag)
            continue
        split = tag.split('-')
        if len(split) != 2 or split[0] not in ["I", "B"]:
            raise TypeError("The encoding schema is not a valid IOB type.")
        if split[0] == "B":
            output.append(tag)
            continue
        elif i == 0 or tags[i - 1] == "O":
            output.append("B" + tag[1:])
        elif tags[i - 1][1:] == tag[1:]:
            output.append(tag)
            continue
        else:
            output.append("B" + tag[1:])
    return output

def iob2_to_bioes(tags: List[str]) -> List[str]:
    """iob2 -> bioes"""
    output = []
    for i, tag in enumerate(tags):
        if tag == "O":
            output.append(tag)
        else:
            split = tag.split("-")[0]
            if split == "B":
                if i + 1 != len(tags) and tags[i + 1].split("-")0] == "I":
                    output.append(tag)
                else:
                    output.append(tag.replace("B-", "S-"))
            elif split == "I":
                if i + 1 < len(tags) and tags[i + 1].split("-")[0] == "I":
                    output.append(tag)
                else:
                    output.append(tag.replace("I-", "E-"))
            else:
                raise TypeError("Invalid IOB format")
    return output

def iob2_to_bmeso(tags: List[str]) -> List[str]:
    """iob2 -> bmeso"""
    output = []
    for i, tag in enumerate(tags):
        if tag == "O":
            output.append(tag)
        else:
            split = tag.split("-")[0]
            if split == "B":
                if i + 1 != len(tags) and tags[i + 1].split("-")[0] == "I":
                    output.append(tag)
                else:
                    output.append(tag.replace("B-", "S-"))
            elif split == "I":
                if i + 1 < len(tags) and tags[i + 1].split("-")[0] == "I":
                    output.append(tag.replace("I-", "M-"))
                else:
                    output.append(tag.replace("I-", "E-"))
            else:
                raise TypeError("Invalid IOB format." + "".join(tags))
    return output


def bmes_to_iob2(tags: List[str]) -> List[str]:
    return spans_to_iob2(bmes_to_spans(tags), len(tags))

def bmeso_to_iob2(tags: List[str]) -> List[str]:
    return spans_to_iob2(bmeso_to_spans(tags), len(tags))

def tags_to_tags(tags: List[str], src: EntScheme, tgt: EntScheme) -> List[str]:
    if src == tgt:
        return tags
    else:
        spans = ent_tags_to_spans(src)(tags)
        return spans_to_iob2(spans, len(tags))

def infer_scheme_from_vocab(tag_vocab: Mapping) -> EntScheme:
    tag_set = set()
    unk_token: Optional[str] = "<unk>"
    pad_token: Optional[str] = "<pad>"
    for tag in tag_vocab.values():
        if tag in (unk_token, pad_token):
            continue
        tag = tag[:1].upper()
        tag_set.add(tag)

    bmes_tag_set = set("BMES")
    if tag_set == bmes_tag_set: 
        return EntScheme.BMES
    bio_tag_set = set("BIO")
    if tag_set == bio_tag_set:
        return EntScheme.IOB2
    bmeso_tag_set = set("BMESO")
    if tag_set == bmeso_to_spans:
        return EntScheme.BMESO
    biose_tag_set = set("BIOES")
    if tag_set == biose_tag_set:
        return EntScheme.BIOES
    raise RuntimeError("Can't infer NER scheme")

def check_scheme_from_vocab(tag_vocab: Mapping, scheme: EntScheme):
    tag_set = set()
    unk_token: Optional[str] = "<unk>"
    pad_token: Optional[str] = "<pad>"
    for tag in tag_vocab.values():
        if tag in (unk_token, pad_token):
            continue
        tag = tag[:1].upper()
        tag_set.add(tag)
    
    tags = str(scheme)
    for tag in tag_set:
        assert tag in tags, f'{tag} is not a valid tag in encoding type:{scheme}.\
        Please check your ' f'encoding_type'
        tags = tags.replace(tag, "") #　删除该值
    if tags: # 如果不为空，说明出现了未使用的tag
        logger.warning(
            f"Tag:{tags} in encoding type:{scheme} is not presented in your Vocabulary.\
            Check your " "encoding_type"
        )

