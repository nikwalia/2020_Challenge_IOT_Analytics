import stat_helper
import hdf_helper
import pandas as pd
import multiprocessing
import time

def write_csv(channel_id):
    df = hdf_helper.get_channel_data(channel_id)
    df.to_csv('dat/dat_' + str(channel_id) + '.csv')

if __name__ == '__main__':
    
    processes = []
    channel_names = ['ch_' + str(i) for i in range(1, 143)]
    print(channel_names)

    print(multiprocessing.cpu_count())

    cur_channel = 16
    for i in range(multiprocessing.cpu_count()):
        p = multiprocessing.Process(target = write_csv, args = (channel_names[cur_channel],))
        p.start()
        print(i, channel_names[cur_channel])
        processes.append(p)
        cur_channel += 1

                