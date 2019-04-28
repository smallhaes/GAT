def var_detail(variable,all_global_vars=globals()):
    '''

    :param variable: np.ndarray, np.matrix
    :return:
    '''
    # print('===============================')
    # for x in all_global_vars:
    #     print(x)
    # print('===============================\n\n')
    l = [name for name in all_global_vars if all_global_vars[name] is variable]
    if len(l) == 0:
        print('===============================')
        print('没有这个变量')
        print('===============================\n\n')
    else:
        var_name = l[0]
        print('=================================')
        print('%s\n'%var_name,variable)
        print('%s.type'%var_name,type(variable))
        print('%s.shape'%var_name,variable.shape)
        print('%s.size'%var_name,variable.size)
        print('%s.ndim'%var_name,variable.ndim)
        print('%s.dtype'%var_name,variable.dtype)
        print('=================================\n\n')