import numpy as np
# acc per sentence
# acc per token
# mrr per token
# mrr per sentence
# entropy per token
# entorpy per sentnece

def stats(y_hat, targets, metrics):
  acc_token, mrr_token, acc10_token = metrics

  # accuracy
  pred_idxs = y_hat.argmax(axis=1)
  acc = np.sum(targets==pred_idxs)
  acc_token+=acc

  # mrr 
  ranks = (np.negative(y_hat)).argsort(axis=1)
  ranks_of_best = np.where(ranks==targets.reshape(-1,1))[1]
  recip_ranks = 1.0 / np.add(ranks_of_best,1)

  # acc @ k
  acc10 = np.where(ranks_of_best<10)[0]
  acc10_token+=len(acc10)

  acc_token+=acc
  mrr_token+=np.sum(recip_ranks)
  metrics = [acc_token, mrr_token, acc10_token]
  return metrics 
