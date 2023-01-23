from numpy import ones


def getstartpoints(first_row, first_col, matrix):
    first_col = round(first_col)
    first_row = round(first_row)
    maxlen = matrix.shape[1]
    edgerows = []
    flag = 0

    # 从第一起始点向下遍历
    # 用flag避免粘连情况
    for i in range(first_row, maxlen):
        if matrix[i, first_col] != 0 and flag == 0:
            edgerows.append(i)
            flag = 1
        elif matrix[i, first_col] == 0:
            flag = 0
    
    # 第一起始点也是起始点
    rows = [first_row]

    # 每两个边点取一个中点加入起始点队列
    for i in range(0, len(edgerows)-1):
        rows.append(round((edgerows[i]+edgerows[i+1])/2))
    
    # 行坐标全部与第一起始点一致
    cols = ones((1, len(rows))) * first_col

    return [rows, cols[0]]
    

