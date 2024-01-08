from .glue import (
    GLUEcola,
    GLUEmnli,
    GLUEmrpc,
    GLUEqnli,
    GLUEqqp,
    GLUErte,
    GLUEsst2,
    GLUEstsb,
    GLUEwnli
)
from .tony_zhao import (
    TonyZhaoTREC,
    TonyZhaoSST2,
    TonyZhaoAGNEWS,
    TonyZhaoDBPEDIA
)


dataset2class = {
    "glue/cola": GLUEcola,
    "glue/mnli": GLUEmnli,
    "glue/mrpc": GLUEmrpc,
    "glue/qnli": GLUEqnli,
    "glue/qqp": GLUEqqp,
    "glue/rte": GLUErte,
    "glue/sst2": GLUEsst2,
    "glue/stsb": GLUEstsb,
    "glue/wnli": GLUEwnli,

    "tony_zhao/trec": TonyZhaoTREC,
    "tony_zhao/sst2": TonyZhaoSST2,
    "tony_zhao/agnews": TonyZhaoAGNEWS,
    "tony_zhao/dbpedia": TonyZhaoDBPEDIA,
}