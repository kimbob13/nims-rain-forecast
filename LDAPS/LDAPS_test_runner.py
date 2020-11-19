import os
import sys

MONTH_DAY = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

if __name__ == '__main__':
    year = int(sys.argv[1])
    if year == 2020:
        MONTH_DAY[2] = 29

    start_month = int(sys.argv[2])
    end_month = int(sys.argv[3])
    month_list = list(range(start_month, end_month + 1))

    for month in month_list:
        for day in range(1, MONTH_DAY[month] + 1):
            cmd = 'python3 LDAPS_test.py --test_time={:4d}{:02d}{:02d}'.format(year, month, day)
            print('Running...', cmd)
            os.system(cmd)
            print()
