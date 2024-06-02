from hmmlearn import hmm

def getHmmModel(model_name:str, n_components: int, cov_type:str, n_iter:int):
  if model_name == "GaussianHMM" :
    return hmm.GaussianHMM(n_components=n_components, covariance_type=cov_type,n_iter=n_iter)
  elif model_name == "GMMHMM" :
    return hmm.GMMHMM(n_components=n_components, covariance_type=cov_type,n_iter=n_iter)
  if model_name == "MultinomialHMM" :
    return hmm.MultinomialHMM(n_components=n_components, covariance_type=cov_type,n_iter=n_iter)
  else :
    return "invalid Name"