# 34118 - area introduced  and 31452 - average price
# 1 - monetary value of sold flats for a quarter

import pandas as pd

import main2
# main2.get_data('34118')
[area, titles_area] = main2.load_data('34118')
PERIOD = ['январь', 'февраль', 'март', 'апрель', 'май', 'июнь',
          'июль', 'август', 'сентябрь' 'октябрь', 'ноябрь', 'декабрь']
area = area[area['PERIOD'].isin(PERIOD)]
area.PERIOD = area.PERIOD.astype('category')
area.PERIOD.cat.set_categories(PERIOD)

area.sort_values(['s_OKATO', 'TIME', 'PERIOD'])
# associate technical names with human-readable
col_names = ['TIME', 'PERIOD', 's_OKATO', 's_OKATO_id_x', 's_mosh', 'EI', 'VALUE']
nice_names = ['Year', 'Period', 'Federal District', 'Federal District (id)', 'Type of Building',
              'Unit of Area', 'Area Introduced']
name_dict = dict(zip(col_names, nice_names))

# Replace technical names with human-readable
area = area.rename(columns=name_dict)

area_pivot = area.pivot(index=['Federal District', 'Type of Building'], columns=['Year', 'Period'], values='Area Introduced')
area_pivot.to_csv('Area Introduced Monthly.csv', encoding='utf-8')

print(area_pivot)






#
# # Filter values
# s_mosh = ['Жилые здания многоквартирные',]                                                         # 34118
# s_vidryn = ['Первичный рынок жилья',]
# s_OKATO = ['Центральный федеральный округ', 'Северо-Западный федеральный округ',
#            'Южный федеральный округ (с 29.07.2016)', 'Северо-Кавказский федеральный округ',
#            'Приволжский федеральный округ', 'Уральский федеральный округ',
#            'Сибирский федеральный округ', 'Дальневосточный федеральный округ']
#
# S_TIPKVARTIR = ['Все типы квартир',]
# PERIOD = ['I квартал', 'II квартал', 'III квартал', 'IV квартал']
# years = [2019, 2020, 2021]
#
# # Filtering of Areas
# area = area.set_index(['s_OKATO', 'TIME', 'PERIOD'])
#
# # Calculate values for the 1st quarter of 2019
# quarter_index = pd.MultiIndex.from_arrays([s_OKATO + s_OKATO, [2019, 2019, 2019, 2019, 2019, 2019, 2019, 2019,
#                                                      2020, 2020, 2020, 2020, 2020, 2020, 2020, 2020],
#                                            ['I квартал', 'I квартал', 'I квартал', 'I квартал',
#                                             'I квартал', 'I квартал','I квартал', 'I квартал',
#                                             'I квартал', 'I квартал', 'I квартал', 'I квартал',
#                                             'I квартал', 'I квартал','I квартал', 'I квартал']])
# first_quarter = pd.DataFrame(index = quarter_index, columns=['VALUE',])
# first_quarter = first_quarter.sort_index()
# for district in s_OKATO:
#     for year in [2019, 2020]:
#         temp_quarter = area.loc[(district, year, ['январь', 'февраль', 'март']), 'VALUE']
#         first_quarter.loc[(district, year,'I квартал'), 'VALUE'] = temp_quarter.sum()
#
# area = pd.concat([area, first_quarter])
# area = area.loc[(s_OKATO, years, PERIOD)]
# area = area.reset_index()
# price_area = price.merge(area, left_on=['s_OKATO', 'TIME', 'PERIOD'], right_on=['s_OKATO', 'TIME', 'PERIOD'])
#
# # Calculate monetary value
#
# associate technical names with human-readable
# col_names = ['TIME', 'PERIOD', 's_OKATO', 's_OKATO_id_x', 's_vidryn', 's_mosh', 'EI_x', 'EI_y', 'VALUE_x', 'VALUE_y',
#              'S_TIPKVARTIR']
# nice_names = ['Year', 'Period', 'Federal District', 'Federal District (id)', 'Type of Market', 'Type of Building',
#               'Unit of Price', 'Unit of Area', 'Average Price', 'Area Introduced', 'Type of Flats']
# name_dict = dict(zip(col_names, nice_names))
#
# # Replace technical names with human-readable
# price_area = price_area.rename(columns=name_dict)
# area = area.rename(columns=name_dict)
# price = price.rename(columns=name_dict)
#
# price_area.to_csv('Monetary Value Report.csv')
#
# # Create pivot-tables
# price_area = price_area.pivot(index='Federal District', columns=['Year', 'Period'], values="Monetary Value in millions ₽")
# price_pivot = price.pivot(index='Federal District', columns=['Year', 'Period'], values='VALUE')
# area_pivot = area.pivot(index='Federal District', columns=['Year', 'Period'], values='VALUE')
#
# inp = input('Would you like to add separate tables for price and area? pr/ar/both/no ')
# inp_d = {'both': [price, area], 'pr': [price,], 'ar': [area,], 'no': []}
# price.name = 'price'
# area.name = 'area'
# for i in inp_d[inp]:
#     i.to_csv(i.name + '.csv')
#     print(i)
#     # price.to_csv('Average Price of Square Meter.csv')
#     # area.to_csv('Area Introduced.csv')
#
# print(price_area)

# print('Area (34118): \n', area)
# print('Price (31452): \n', price)

# s_OKATO_id = ['030', '031', '040', '038', '033', '034', '041', '042']
# region_dict = {s_OKATO[i]: s_OKATO_id[i] for i in range(len(s_OKATO))}