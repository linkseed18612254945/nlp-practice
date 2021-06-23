import datetime
import time

TIME_FORMAT = '%Y-%m-%d %H:%M:%S'


def time_string_to_timestamp(time_string):
    """
    将时间字符串装换为时间戳
    :param time_string:  时间字符串，格式为TIME_FORMAT格式
    :return: float， 秒级时间戳
    """
    return time.mktime(time.strptime(time_string, TIME_FORMAT))


def time_format_change(time_string, format1, format2):
    """
    将时间字符串装换为时间戳
    :param time_string:  时间字符串，格式为TIME_FORMAT格式
    :return: float， 秒级时间戳
    """
    timestamp = time.mktime(time.strptime(time_string, format1))
    new_time_string = time.strftime(format2, time.localtime(timestamp))
    return new_time_string


def timestamp_to_time_string(timestamp):
    """
    时间戳转换为时间字符串
    :param timestamp: float, 秒级时间戳
    :return: string，时间字符串，格式为TIME_FORMAT格式
    """
    time_local = time.localtime(timestamp)
    return time.strftime(TIME_FORMAT, time_local)


def get_now_time():
    """
    获得当前时间的时间字符串
    :return: string, 时间字符串，格式为TIME_FORMAT格式
    """
    return datetime.datetime.now().strftime(TIME_FORMAT)


def gap_days(gap):
    """
    获取当前时间和指定天数之前的时间
    :param gap: int， 天数
    :return start: string， gap天前的时间
    :return end: string，当前时间
    """
    now = datetime.datetime.now()
    end = now.strftime(TIME_FORMAT)
    start = (now - datetime.timedelta(gap)).strftime(TIME_FORMAT)
    return start, end


def get_range_time(start, gap_day):
    """
    获取从起始时间算起到指定天数之后的时间
    :param start: string， 指定的起始时间
    :param gap_day: int, 向后推算天数
    :return: string， 开始时间和结束时间字符串
    """
    start_time = datetime.datetime.strptime(start, TIME_FORMAT)
    end = (start_time - datetime.timedelta(gap_day)).strftime(TIME_FORMAT)
    return start, end


if __name__ == '__main__':
    f2 = TIME_FORMAT
    f1 = '%Y/%m/%d'
    c = time_format_change('2020/5/27', f1, f2)
    print(c)