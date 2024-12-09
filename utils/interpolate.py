from activation_dataset import loadActivationCsv

def nonlinearFnc(inp, data):
    x, y = data
    
    break_flag = False
    for i, val in enumerate(x):
        if val > inp:
            break_flag = True
            break
        
    if i == 0:
        ret = y[0]
    elif not break_flag:
        ret = y[len(y)-1]
    else:
        ratio = (inp - x[i-1]) / (x[i] - x[i-1])
        ret = y[i-1] + (y[i] - y[i-1])*ratio
        
    return ret

