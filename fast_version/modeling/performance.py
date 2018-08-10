import cupy as cp
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
  acc = cp.sum(targets==pred_idxs)
  acc_token+=acc

  # mrr 
  ranks = (cp.negative(y_hat)).argsort(axis=1)
  ranks_of_best = cp.where(ranks==targets.reshape(-1,1))[1]
  recip_ranks = 1.0 / cp.add(ranks_of_best,1)

  # acc @ k
  acc10 = cp.where(ranks_of_best<10)[0]
  acc10_token+=len(acc10)

  acc_token+=acc
  mrr_token+=cp.sum(recip_ranks)
  metrics = [acc_token, mrr_token, acc10_token]
  return metrics 
