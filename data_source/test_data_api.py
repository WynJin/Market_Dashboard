# 这是一个示例 Python 脚本。

# 按 Shift+F10 执行或将其替换为您的代码。
# 按 双击 Shift 在所有地方搜索类、文件、工具窗口、操作和设置。
import akshare as ak


import tushare as ts
import tushare.pro.client as _client
_client.DataApi._DataApi__http_url = 'http://api.quicksync.cn'
pro = ts.pro_api('3603bca126bf59588a0545dd94e21089b5330801a963f7b8fb670551')


def print_hi(name):


    # 在下面的代码行中使用断点来调试脚本。
    print(f'Hi, {name}')  # 按 Ctrl+F8 切换断点。

def test_akshare():

    search_df = ak.search("分时", limit=5)
    print(search_df)


def test_tushare():

    # 测试 rt_idx_min
    try:
        df = pro.rt_min(ts_code='600000.SH', freq='5MIN')
        print(f'rt_min 成功: {len(df)} 行')
        print(df.columns.tolist())
        print(df.head(3))
    except Exception as e:
        print(f'rt_min 失败: {type(e).__name__}: {str(e)[:100]}')

    # 测试 rt_idx_min
    try:
        df = pro.rt_idx_min(ts_code='000001.SH', freq='5MIN')
        print(f'rt_idx_min 成功: {len(df)} 行')
        print(df.columns.tolist())
        print(df.head(3))
    except Exception as e:
        print(f'rt_idx_min 失败: {type(e).__name__}: {str(e)[:100]}')

    # 测试 rt_idx_min_daily
    try:
        df = pro.rt_idx_min_daily(ts_code='000001.SH', freq='5MIN')
        print(f'rt_idx_min_daily 成功: {len(df)} 行')
    except Exception as e:
        print(f'rt_idx_min_daily 失败: {type(e).__name__}')

    # 测试 rt_sw_k
    try:
        df = pro.rt_sw_k(ts_code='801053.SI')
        print(f'rt_sw_k 成功: {len(df)} 行')
    except Exception as e:
        print(f'rt_sw_k 失败: {type(e).__name__}')

    # 测试 ths_index (板块列表)
    try:
        df = pro.ths_index(type='I')
        print(f'ths_index 行业: {len(df)} 行')
        print(df.columns.tolist())
    except Exception as e:
        print(f'ths_index 失败: {type(e).__name__}: {str(e)[:60]}')


# 按装订区域中的绿色按钮以运行脚本。
if __name__ == '__main__':
    # print_hi('PyCharm')
    test_tushare()


