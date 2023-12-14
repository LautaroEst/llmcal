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


dataset2class = {
    "glue/cola": GLUEcola,
    "glue/mnli": GLUEmnli,
    "glue/mrpc": GLUEmrpc,
    "glue/qnli": GLUEqnli,
    "glue/qqp": GLUEqqp,
    "glue/rte": GLUErte,
    "glue/sst2": GLUEsst2,
    "glue/stsb": GLUEstsb,
    "glue/wnli": GLUEwnli
}