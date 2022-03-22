def are_data_same(a,b):
    for i in range(a,b):
        if a[i] != b[i]: return False
    return True

def load_formatted_data(infile):
    dataset = np.loadtxt(infile, comments=None, encoding='utf-8',
                         dtype= 'float')
    return dataset

file1 = sys.argv[1] 
file2 = sys.argv[2] 
a = load_formatted_data(file1)
b= load_formatted_data(file2)

are_data_same(a,b)