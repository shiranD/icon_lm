from torch.autograd import Variable
import numpy as np
from embedder import index2embed
from performance_reg import stats

def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if isinstance(h,Variable):
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)

def get_batch(source, i, bptt, evaluation=False):
    seq_len = min(bptt, len(source) - 1 - i)
    data = Variable(source[i:i+seq_len], volatile=evaluation)
    target = Variable(source[i+1:i+1+seq_len].view(-1))
    return data, target

def feedfwd(model, data_source, embedict, bptt, eval_batch_size, dictionary, emsize, criterion):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    ntokens = len(dictionary)
    hidden = model.init_hidden(eval_batch_size)
    total_loss = 0
    acc_token = np.float64(0)
    acc10_token = np.float64(0)
    mrr_token = np.float64(0)
    ln = np.int64(0)
    metrics = [acc_token, mrr_token, acc10_token]
    for i in range(0, data_source.size(0) - 1, bptt):
        data, targets = get_batch(data_source, i, bptt, evaluation=True)
        data = index2embed(data, embedict, dictionary, emsize)
        output, hidden = model(data, hidden)
        output_flat = output.view(-1, ntokens)
        pred = output_flat.data.cpu().numpy()        
        refers = targets.data.cpu().numpy()
        # convert to cupy arrays and send to performance
        pred = np.array(pred)
        refers = np.array(refers)
        ln+=len(refers)
        metrics = stats(pred, refers, metrics)
        hidden = repackage_hidden(hidden)
    return metrics[0] / ln*100 , metrics[1] / ln, metrics[2] / ln*100

