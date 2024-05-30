class EmbedsReduced:
  """ Use this for pre-calculated reduced embeddings """
  def __init__(self, reduced_embeddings):
    self.reduced_embeddings = reduced_embeddings

  def fit(self, X):
    return self

  def transform(self, X):
    return self.reduced_embeddings

def create_preumap(reduced_embeddings):
    return EmbedsReduced(reduced_embeddings)