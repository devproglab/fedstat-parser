# 34118 - area introduced  and 31452 - average price
# 1 - monetary value of sold flats for a quarter
import main2

id = 34118
[area, titles_area] = main2.load_data('34118')
[price, titles_price] = main2.load_data('31452')
# Filter values
s_mosh = 'Жилые здания многоквартирные'
s_vidryn = 'Первичный рынок жилья'
s_OKATO = ['Центральный федеральный округ', 'Северо-Западный федеральный округ',
           'Южный федеральный округ (с 29.07.2016)', 'Северо-Кавказский федеральный округ',
           'Приволжский федеральный округ', 'Уральский федеральный округ',
           'Сибирский федеральный округ', 'Дальневосточный федеральный округ']
# s_OKATO_id = ['030', '031', '040', '038', '033', '034', '041', '042']
# region_dict = {s_OKATO[i]: s_OKATO_id[i] for i in range(len(s_OKATO))}
S_TIPKVARTIR = ['Все типы квартир',]
PERIOD = ['I квартал', 'II квартал', 'III квартал', 'IV квартал']
years = [2019, 2020, 2021]
# area = area[area['S_TIPKVARTIR'].isin(S_TIPKVARTIR)]
# area['s_OKATO'] = s_OKATO

s_mosh = 'Жилые здания многоквартирные'

print('Area (34118): \n', area["s_OKATO"], '\n', area.dtypes)
print('Price (31452): \n', price.head(),'\n', price.dtypes)