import sys
import pandas as pd 

if len(sys.argv) < 3:
    print("Error: two csv files are needed, old first new second")
    exit(1)

old_perf_file = str(sys.argv[1])
new_perf_file = str(sys.argv[2])

verbose = False
if len(sys.argv) == 4:
    verbose = bool(sys.argv[3])

print("Comparing {} vs {}".format(old_perf_file, new_perf_file))

df_old = pd.read_csv (old_perf_file, sep=';', header=None, index_col=0)
df_new = pd.read_csv (new_perf_file, sep=';', header=None, index_col=0)

for index, row in df_new.iterrows():
    if index in df_old.index:
        if verbose:
            print("New: ",row[1], row[2], row[3])
            print("Old: ",df_old.loc[index][1], df_old.loc[index][2], df_old.loc[index][3])

        print(index, end=': ')
        if abs(row[1] - df_old.loc[index][1]) < df_old.loc[index][1]*0.1: 
            print("similar performance")
        elif (row[1] < df_old.loc[index][1]): 
            print('\x1b[3;30;42m' + 'faster' + '\x1b[0m')
        elif (row[1] > df_old.loc[index][1]): 
            if row[1] > df_old.loc[index][1] + df_old.loc[index][1] * 0.5 :
                print('\x1b[3;30;41m' + 'WAY SLOWER' + '\x1b[0m')
            else:
                print('\x1b[3;30;41m' + 'slower' + '\x1b[0m')


