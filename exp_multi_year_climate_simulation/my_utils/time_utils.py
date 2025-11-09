import datetime

def get_date(year,day):
    # 输入的字符串类型的年和日转换为整型
    year = int(year)
    day  = int(day)
    first_day = datetime.datetime(year,1,1)
    wanted_day=first_day+datetime.timedelta(day)
    wanted_day=datetime.datetime.strftime(wanted_day,'%Y%m%d')
    return wanted_day

if __name__ == '__main__':
    ics = np.arange(0, stop, params.decorrelation_time)
    date_str = get_date(2012, 321)
    print(date_str)
