import xlrd
import numpy as np
import pandas as pd


def matrix(path):
    table = xlrd.open_workbook(path).sheets()[0]
    row = table.nrows
    col = table.ncols
    datamatrix = np.zeros((row, col))
    for x in range(col):
        try:
            cols = np.matrix(table.col_values(x))
            datamatrix[:, x] = cols
        except:
            print(x)

    # print(datamatrix.shape)
    return datamatrix


def csv_to_Matrix(path):
    x_Matrix = pd.read_csv(path, header=None)
    x_Matrix = np.array(x_Matrix)
    return x_Matrix
