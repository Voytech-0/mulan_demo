# from ltnc import ltnc

def evaluate_embedding_quality(data, embedding, labels):
    """ Compute evaluation metrics on the embedding"""


    # label_tnc = ltnc.LabelTNC(data, embedding, labels, cvm="btw_ch")
    # results = label_tnc.run()
    results = {'lc': 0, 'lt': 0}
    return {
        "Label-Trustworthiness": round(float(results["lt"]), 4),
        "Label-Continuity": round(float(results["lc"]), 4)
    }