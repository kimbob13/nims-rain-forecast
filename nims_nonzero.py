import xarray as xr
import numpy as np

import os
import sys
import pwd

def parse_date(date_argv):
    year = int(date_argv[0])
    month = int(date_argv[1])
    day = int(date_argv[2])
    hour = int(date_argv[3])

    missing_date = ['2016/08/22/18', '2017/02/04/18', '2017/08/08/09',
                    '2017/08/09/06', '2018/02/01/05', '2018/02/01/06']
    input_date = '{}/{:02}/{:02}/{:02}'.format(year, month, day, hour)

    if input_date in missing_date:
        print("You've tried to access missing data!")
        sys.exit(1)

    valid_year = list(range(2009, 2019))
    valid_month = list(range(1, 13))
    valid_hour = list(range(24))

    month_31_day = [1, 3, 5, 7, 8, 10, 12]
    leap_year = [2012, 2016]
    
    if month in month_31_day:
        valid_day = list(range(1, 32))
    elif month == 2:
        if year in leap_year:
            valid_day = list(range(1, 30))
        else:
            valid_day = list(range(1, 29))
    else:
        valid_day = list(range(1, 31))

    if year not in valid_year:
        print('Invalid year, year must be between 2009 and 2018')
    if month not in valid_month:
        print('Invalid month, month must be between 1 and 12')
    if day not in valid_day:
        print('Invalid day, day must be between {} and {}'
               .format(valid_day[0], valid_day[-1]))
    if hour not in valid_hour:
        print('Invalid hour, hour must be between 0 and 23')

    return '{}{:02}{:02}{:02}'.format(year, month, day, hour)

def get_data_path(date):
    for p in pwd.getpwall():
        if p[0].startswith('osilab'):
            data_user = p[0]
            break

    root_dir = os.path.join('/home', data_user, 'hdd/NIMS')
    data_path = os.path.join(root_dir, date[:-2], 'tidx_{}.nc'.format(date))

    return data_path

if __name__ == '__main__':
    assert len(sys.argv) == 6

    date = parse_date(sys.argv[1:5])
    print('date: {}'.format(date))

    dataset = xr.open_dataset(get_data_path(date))
    rain_data = dataset.rain.values.squeeze(0)

    num_searching_nonzero = int(sys.argv[5])
    nonzero_points = np.nonzero(rain_data)

    print('rain_data: {}, nonzero len: {}'
          .format(rain_data.shape, len(nonzero_points[1])))
    for i in range(num_searching_nonzero):
        print('{} - [{}, {}]'
              .format(i, nonzero_points[0][i], nonzero_points[1][i]))
