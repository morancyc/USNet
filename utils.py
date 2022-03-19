import numpy as np


def poly_lr_scheduler(optimizer, init_lr, iter, lr_decay_iter=1, max_iter=500, power=0.9):
    """Polynomial decay of learning rate
		:param init_lr is base learning rate
		:param iter is a current iteration
		:param lr_decay_iter how frequently decay occurs, default is 1
		:param max_iter is number of maximum iterations
		:param power is a polymomial power

	"""
    # if iter % lr_decay_iter or iter > max_iter:
    # 	return optimizer

    lr = init_lr * (1 - iter / max_iter) ** power
    optimizer.param_groups[0]['lr'] = lr
    optimizer.param_groups[1]['lr'] = lr * 0.1
    return lr


def fast_hist(a, b, n):
    '''
	a and b are predict and mask respectively
	n is the number of classes
	'''
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def getScores(conf_matrix):
    if conf_matrix.sum() == 0:
        return 0, 0, 0, 0, 0
    with np.errstate(divide='ignore',invalid='ignore'):
        classpre = np.diag(conf_matrix) / conf_matrix.sum(0).astype(np.float)
        classrecall = np.diag(conf_matrix) / conf_matrix.sum(1).astype(np.float)
        pre = classpre[1]
        recall = classrecall[1]
        F_score = 2 * (recall * pre) / (recall + pre)
        fpr = conf_matrix[0, 1] / np.float(conf_matrix[0, 0] + conf_matrix[0, 1])
        fnr = conf_matrix[1, 0] / np.float(conf_matrix[1, 0] + conf_matrix[1, 1])
    return F_score, pre, recall, fpr, fnr
