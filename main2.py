"""The program is meant to facilitate the uploading of data from fedstat.ru
in order to create reports on indicators chosen by the user.

Pylint rated the code 7.91/10
"""
from sys import exit

try:
    import os.path
    import urllib.request
    import urllib.parse
    import urllib.error
    import json
    import socket
    import re
    import codecs
    import pandas as pd
    from urllib.error import HTTPError, URLError
    from urllib.parse import urlencode
    from urllib.request import Request, urlopen
    import xml.etree.ElementTree as ET
    import time
    import sqlite3
except:
    print(
        'Error: some of the required packages are missing. Please install them.')
    exit()
try:
    conn = sqlite3.connect('data.sqlite')
    cur = conn.cursor()
    conn_str = sqlite3.connect('structure.sqlite')
    cur_str = conn_str.cursor()
except:
    print('Failed to connect to SQL database')


def get_structure(id, force_upd=False):
    """Retreives data structure either from an existing structure database
    or FedStat website.

    Parameters:
    id (str): id of a FedStat indicator.
    force_upd (bool): If True, data in the database gets updated.

    Returns:
    filters (pandas.core.frame.DataFrame): A dataframe with filters
    used by FedStat for their data.

    Knowledge from Coursera courses:
    Course 1 Programming for Everybody for declaring a function,
    using a 'for' loop and conditional 'if' statements.
    Course 2 Python Data Structures for working with different data structures
    (strings, lists, and dictionaries).
    Course 3 Using Python to Access Web Data for retreiving data
    from FedStat website and using regular expressions.
    Course 4 Using Databases with Python for
    creating a database with data structure.
    """
    cur_str.execute('''CREATE TABLE IF NOT EXISTS Structure
        (id INTEGER PRIMARY KEY, database INTEGER, filter_id INTEGER,
         value_id INTEGER, filter_type INTEGER)''')
    cur_str.execute('''CREATE TABLE IF NOT EXISTS Filters
        (filter_id INTEGER PRIMARY KEY, filter_title TEXT)''')
    cur_str.execute('''CREATE TABLE IF NOT EXISTS Filter_values
        (value_id INTEGER PRIMARY KEY, value_title TEXT)''')
    cur_str.execute('''CREATE TABLE IF NOT EXISTS Filter_types
        (filter_type INTEGER PRIMARY KEY AUTOINCREMENT,
         filter_type_title TEXT UNIQUE)''')
    # try to get structure
    cur_str.execute('SELECT id FROM Structure WHERE database=? LIMIT 1',
                    (str(id), ))
    row = cur_str.fetchone()
    if row is not None and not force_upd:
        print('Reading data structure...')
        clmns = ['filter_id', 'fiter_title', 'value_id', 'value_title',
                 'filter_type']
        cur_str.execute('''SELECT Structure.filter_id, Filters.filter_title,
        Structure.value_id, Filter_values.value_title,
        Filter_types.filter_type_title
        FROM Structure
        JOIN Filters on Structure.filter_id=Filters.filter_id
        JOIN Filter_values on Structure.value_id=Filter_values.value_id
        JOIN Filter_types on Structure.filter_type=Filter_types.filter_type
        WHERE database=?''', (id, ))
        filters = pd.DataFrame(cur_str, columns=clmns)
        filters['filter_id'] = filters.filter_id.astype(str)
        filters['value_id'] = filters.value_id.astype(str)
        print('Data structure retrieved.')
    else:
        # load all possible filter values and codes from the fedstat page
        print('Downloading data structure...')
        url = "https://fedstat.ru/indicator/" + str(id)
        try:
            response = urllib.request.urlopen(url, timeout=300)
        except HTTPError as error:
            print('Data structure not retrieved: ', error, url)
            exit()
        except socket.timeout:
            print('Socket timed out - URL:', url)
            exit()
        except:
            print('Data structure not retrieved. \
                   Check your internet connection')
            exit()
        else:
            data = response.read().decode()
            print('Processing data structure...')
            # extract the necessary part from raw javascript
            # and modify it to be proper json
            pattern = re.compile(r"title:(.+?)\}\);", re.DOTALL | re.MULTILINE)
            try:
                results = pattern.findall(data)[0].strip()
            except:
                print('Wrong dataset id. Please try again')
                user_interface()
            results = '{"title":' + results + '}'
            # transform double-escape unicode characters to proper text
            results = codecs.decode(results, 'unicode-escape')
            results = re.sub("\'", '\"', results)
            results = re.sub(r"\n+\s*(\w+):", r'"\1":', results)
            try:
                results = json.loads(results)
            except:
                print('Data structure retrieving failed')
                results = []
                return results
            # put filter metadata to list
            df2 = list(results["filters"].items())
            # write everything to a single dataframe
            filters = pd.DataFrame(columns=('filter_id',
                                            'filter_title',
                                            'value_id',
                                            'value_title'))
            for i in range(0, len(df2)):
                # extract titles of filters' values
                titles = re.findall("title': '(.+?)'",
                                    str(df2[i][1]["values"]))
                # extract internally used ids of filters' values
                keys = re.findall("'([0-9]+?)': {'title'",
                                  str(df2[i][1]["values"]))
                # merge into df with filter ids and filter titles
                titles = pd.DataFrame(
                    {"value_title": titles,
                     "value_id": keys,
                     "filter_title": len(titles) * [df2[i][1]["title"]],
                     "filter_id": len(titles) * [df2[i][0]]})
                filters = pd.concat([filters, titles])
            # replace "root" value title with dataset name
            filters.loc[filters["filter_id"] == "0", "value_title"] \
                = results["title"]
            # extract internally used filter types and map them to filter ids
            fields = [results["left_columns"], results["top_columns"],
                      results["groups"], results["filterObjectIds"]]
            nfields = [['lineObjectIds'] * len(results["left_columns"]),
                       ['columnObjectIds'] * len(results["top_columns"]),
                       ['columnObjectIds'] * len(results["groups"]),
                       ['lineObjectIds'] * len(results["filterObjectIds"]),
                       ]
            fields = [val for sublist in fields for val in sublist]
            nfields = [val for sublist in nfields for val in sublist]
            layout = pd.DataFrame({"filter_id": fields,
                                   "filter_type": nfields})
            if not any(layout["filter_id"] == "0"):
                layout = pd.concat([layout,
                                    pd.DataFrame({'filter_id': '0',
                                                  'filter_type':
                                                  'filterObjectIds'},
                                                 index=[0])])
            layout["filter_id"] = layout["filter_id"].astype(str)
            filters = pd.merge(filters, layout, how='left')
            # REMOVE LATER: WRITE STRUCTURE TO FILE
            filters['filter_id'] = filters.filter_id.astype(str)
            for _, row in filters.iterrows():
                cur_str.execute('''INSERT OR IGNORE INTO Filters
                (filter_id, filter_title) VALUES (?, ?)''',
                                (str(row['filter_id']), row['filter_title']))
                cur_str.execute('''INSERT OR IGNORE INTO Filter_values
                (value_id, value_title) VALUES (?, ?)''',
                                (str(row['value_id']), row['value_title']))
                cur_str.execute('''INSERT OR IGNORE INTO Filter_types
                (filter_type_title) VALUES (?)''',
                                (str(row['filter_type']),))
                cur_str.execute('SELECT filter_type \
                FROM Filter_types \
                WHERE filter_type_title = ? ',
                                (str(row['filter_type']), ))
                filter_type = cur_str.fetchone()[0]
                cur_str.execute('''INSERT INTO Structure
                (filter_id, value_id, filter_type, database)
                VALUES (?, ?, ?, ?)''',
                                (str(row['filter_id']),
                                 row['value_id'],
                                 filter_type,
                                 id))
            conn_str.commit()
            print('Data structure retrieved.')
    return filters


def query_size(filterdata):
    """Counts how many filters' values there are to determine
    a size of a query needed.

    Parameters:
    filterdata (pandas.core.frame.DataFrame): A dataframe with FedStat filters.

    Returns:
    S (int): The query size, the number of filters.

    Knowledge from Coursera courses:
    Course 1 Programming for Everybody for declaring a function.
    """
    S = filterdata['filter_id'].value_counts().values.prod()
    return S


def make_query(filterdata):
    """Creates a json-style request that will be used to get data from a server.

    Parameters:
    filterdata (pandas.core.frame.DataFrame): A dataframe with FedStat filters.

    Returns:
    query_json (dict): A query in a form of a json-style request.

    Knowledge from Coursera courses:
    Course 1 Programming for Everybody for declaring a function
    and using 'for' loops.
    Course 2 Python Data Structures for working with different data structures
    (strings, lists, and dictionaries).
    """
    # concatenate filter ids and filter value ids for the query
    p = [str(m) + "_" + str(n) for m, n in zip([val[0] for val in filterdata[["filter_id"]].values.tolist()],
                                               [val[0] for val in filterdata[["value_id"]].values.tolist()])]
    query_struct = filterdata.drop_duplicates(subset=["filter_id"]).loc[:, ['filter_type', 'filter_id']]
    # form the query
    meta = filterdata.loc[filterdata["filter_id"] == '0'].values[0]
    query = [('id', meta[2]),
             ('title', meta[3])] + filterdata.drop_duplicates(
                 subset=["filter_id"]).loc[:, ['filter_type', 'filter_id']].to_records(index=None).tolist() \
        + list(zip([['selectedFilterIds'] * len(p)][0], p))
    # format query to json-styled request accepted by the server
    query_json = {}
    for i in query:
        query_json.setdefault(i[0], []).append(i[1])
    return query_json


def parse_sdmx(response, id, nowrite=False):
    """Goes through a datafile received from FedStat and arranges the data
    in an appropriate manner to be written into the database.
    If nowrite=False the data is written into the database,
    if nowrite=True the data is returned in a dataframe.

    Parameters:
    response (urlib.response): An object with data uploaded
    from FedStat website by sending a request to the server.
    id (str): id of a FedStat indicator.
    nowrite (bool): If False, the parsed data is written into the database.

    Returns:
    parsed (pandas.core.frame.DataFrame): A dataframe with parsed data
    from FedStat.

    Knowledge from Coursera courses:
    Course 1 Programming for Everybody for declaring a function,
    using 'for' loops and conditional 'if' statements.
    Course 2 Python Data Structures for working with different data structures
    (strings, lists, and dictionaries).
    Course 3 Using Python to Access Web Data to parse the data retreived
    from FedStat website.
    Course 4 Using Databases with Python for declaring the function
    (week 1 material about OOP).
    """
    # decode .sdmx data parse document tree
    print('Reading data...')
    data = response.read().decode("utf-8")
    if len(data) == 0:
        return
    print('Processing data...')
    try:
        tree = ET.fromstring(data)
    except:
        return
    # define namespace corrsepondences to correctly parse xml data
    ns = {'common': 'http://www.SDMX.org/resources/SDMXML/schemas/v1_0/common',
          'compact': 'http://www.SDMX.org/resources/SDMXML/schemas/v1_0/compact',
          'cross': 'http://www.SDMX.org/resources/SDMXML/schemas/v1_0/cross',
          'generic': 'http://www.SDMX.org/resources/SDMXML/schemas/v1_0/generic',
          'query': 'http://www.SDMX.org/resources/SDMXML/schemas/v1_0/query',
          'structure': 'http://www.SDMX.org/resources/SDMXML/schemas/v1_0/structure',
          'utility': 'http://www.SDMX.org/resources/SDMXML/schemas/v1_0/utility',
          'xsi': 'http://www.w3.org/2001/XMLSchema-instance',
          '': 'http://www.SDMX.org/resources/SDMXML/schemas/v1_0/message'
          }
    # get internal filter ids
    fields_id = list()
    for node in tree.findall('CodeLists/structure:CodeList/[@id]', ns):
        fields_id.append(node.attrib["id"])
    # get filter names
    fields_title = list()
    for node in tree.findall('CodeLists/structure:CodeList/structure:Name', ns):
        fields_title.append(node.text)
    # get internal filter value ids
    fields_codes = list()
    for _, fields_id_item in enumerate(fields_id):
        loc = 'CodeLists/structure:CodeList/[@id="' \
            + fields_id_item + '"]/structure:Code'
        temp = list()
        for node in tree.findall(loc, ns):
            temp.append(node.attrib["value"])
        fields_codes.append(temp)
    # get filter value names
    fields_values = list()
    for _, fields_id_item in enumerate(fields_id):
        loc = 'CodeLists/structure:CodeList/[@id="' \
            + fields_id_item + '"]/structure:Code/structure:Description'
        temp = list()
        for node in tree.findall(loc, ns):
            temp.append(node.text)
        fields_values.append(temp)
    # rename possible duplicates caused by internal rosstat code changes
    # for Sibersky & Dalnevostochny Federal Districts
    idx = fields_id.index('s_OKATO')
    fields_codes[idx] = [w.replace('035', '041') for w in fields_codes[idx]]
    fields_codes[idx] = [w.replace('036', '042') for w in fields_codes[idx]]
    # extract data from the xml structure
    order = list()
    # get all possible values for conventional filters
    for node in tree.findall('DataSet/generic:Series[generic:SeriesKey]', ns)[0]:
        for child in node.findall('generic:Value/[@concept]', ns):
            feature = child.attrib["concept"]
            order.append(feature)
    # now we know names of columns
    colnames = order + ["TIME", "VALUE"]
    # get all possible values for other filter
    temp = list()
    for node in tree.findall('DataSet/generic:Series/generic:Attributes', ns):
        for child in node.findall('generic:Value/[@concept]', ns):
            feature = child.attrib["concept"]
            value = child.attrib["value"]
            temp.append((feature, value))
    temp = set(temp)
    result = {}
    for i in temp:
        result.setdefault(i[0], []).append(i[1])
    for i in range(0, len(list(result.keys()))):
        fields_id.append(list(result.keys())[i])
        fields_values.append(list(result.values())[i])
        if list(result.keys())[i] == 'PERIOD':
            fields_title.append('Период')
        elif list(result.keys())[i] == 'EI':
            fields_title.append('Единица измерения')
    temp_df = list()
    for node in tree.findall('DataSet/generic:Series', ns):
        temp_ds = []
        count = 0
        for child in node.findall('generic:SeriesKey//', ns):
            v = child.attrib["value"]
            if v == '035':
                v = '041'
            if v == '036':
                v = '042'
            temp_ds.append(v)
            count += 1
        for child in node.findall('generic:Attributes//', ns):
            temp_ds.append(child.attrib["value"])
        for child in node.findall('generic:Obs/generic:Time', ns):
            temp_ds.append(int(child.text))
        for child in node.findall('generic:Obs/generic:ObsValue', ns):
            value = re.sub(',', '.', child.attrib["value"])
            temp_ds.append(float(value))
        temp_df.append(temp_ds)
    if not nowrite:
        write_db(id, order, colnames, fields_id, fields_title, fields_values,
                 fields_codes, temp_df)
    else:
        parsed = pd.DataFrame.from_records(temp_df, columns=colnames)
        return parsed


def write_db(id, order, colnames, fields_id, fields_title, fields_values,
             fields_codes, df):
    """Writes parsed data into the database. Each FedStat indicator has
    its own table Data + indicator id. Names of filters are written into
    their own table.

    Parameters:
    id (str): id of a FedStat indicator.
    order (list): A list of column names in proper order.
    colnames (list): A list with future column names.
    order columns plus time and value columns
    fields_id (list): A list with filter ids.
    fields_title (list): A list with filter titles.
    fields_values (list): A list with filter value names.
    fields_codes (list): A list with filter value ids.
    df (list): A list whose elements are also lists.
    Stores data that will be written into the database

    Results in data being written into the database.

    Knowledge from Coursera courses:
    Course 1 Programming for Everybody for declaring a function,
    using 'for' loops and conditional 'if' statements.
    Course 2 Python Data Structures for working with different data structures
    (strings, lists, and tuples).
    Course 4 Using Databases with Python for creating and working with database tables.
    """
    # begin with table creation
    script = 'CREATE TABLE IF NOT EXISTS Data' + str(id) \
        + ' (id INTEGER PRIMARY KEY AUTOINCREMENT UNIQUE, Time INTEGER, \
        Value REAL)'
    cur.executescript(script)
    cur.execute('SELECT max(id) FROM Data' + str(id))
    try:
        r = cur.fetchone()[0]
    except:
        r = None
    for i in order:
        script = 'CREATE TABLE IF NOT EXISTS ' + str(i) + \
                 ' (id INTEGER PRIMARY KEY AUTOINCREMENT UNIQUE, \
                 fed_id TEXT UNIQUE, fed_title TEXT UNIQUE)'
        cur.executescript(script)
        if r is None:
            script = 'ALTER TABLE Data' + str(id) + ' ADD ' + str(i) + ' INTEGER'
            cur.execute(script)
    for i in range(0, len(fields_id)):
        # save filtername we are working with
        c = fields_id[i]
        # map filter ids to human-readable titles
        script = 'CREATE TABLE IF NOT EXISTS Filternames \
                 (id INTEGER PRIMARY KEY AUTOINCREMENT UNIQUE, \
                 field_id TEXT UNIQUE, field_title TEXT)'
        cur.executescript(script)
        script = 'INSERT OR IGNORE INTO Filternames (field_id, field_title) ' \
                 'VALUES ( ?, ? )'
        cur.execute(script, (fields_id[i], fields_title[i]))
        # add some predetermined names as well
        script = 'INSERT OR IGNORE INTO Filternames (field_id, field_title) ' \
                 'VALUES ("TIME", "Год")'
        cur.execute(script)
        script = 'INSERT OR IGNORE INTO Filternames (field_id, field_title) ' \
                 'VALUES ("VALUE", "Значение")'
        cur.execute(script)

        # iterate trough all values for each filter
        for j in range(0, len(fields_values[i])):
            # add filter decoders into DB
            script = 'INSERT OR IGNORE INTO ' + str(c) \
              + ' (fed_id, fed_title) VALUES ( ?, ? )'
            if c not in ['EI', 'PERIOD']:
                cur.execute(script,
                            (fields_codes[i][j], str(fields_values[i][j])))
            else:
                cur.execute(script, (None, str(fields_values[i][j])))
    conn.commit()
    for i in df:
        count = 0
        temp_ds = list()
        for j in order:
            if j not in ['EI', 'PERIOD']:
                # build table relations
                script = 'SELECT id FROM ' + j + ' WHERE fed_id=?'
            else:
                # build table relations
                script = 'SELECT id FROM ' + j + ' WHERE fed_title=?'
            cur.execute(script, (i[count],))
            for item in cur:
                temp_ds.append(item[0])
            count += 1
        # Create SQLite command for any number of columns
        for z in i[-(len(colnames)-count):]:
            temp_ds.append(z)
        script = 'INSERT INTO Data' + str(id) + ' ('
        left = ')  VALUES ('
        for col in colnames:
            script = script + col + ' , '
            left = left + '?,'
        script = script.rstrip(' , ')
        left = left.rstrip(',') + ')'
        script = script + left
        row_to_upload = tuple(temp_ds)
        cur.execute(script, row_to_upload)
    conn.commit()
    cols = ''
    for item in colnames:
        cols = cols + item + ';'
    cur.execute('CREATE TABLE IF NOT EXISTS Metadata \
    (id INTEGER PRIMARY KEY AUTOINCREMENT UNIQUE, \
    dataset INTEGER UNIQUE, columns TEXT, periods TEXT)')
    # if metadata for this dataset already exists, do nothing
    # else append new row
    cur.execute('INSERT OR IGNORE INTO Metadata \
    (dataset, columns) VALUES (?, ?)', (id, cols))
    conn.commit()
    parsed = load_data(id)[0]
    last_upd = set(list(tuple(x) for x in parsed.loc[:, ["TIME", "PERIOD"]].values))
    cur.execute('SELECT id FROM Metadata WHERE dataset = ?', (str(id),))
    row = cur.fetchone()
    for item in row:
        meta_id = item
    cur.execute('UPDATE Metadata SET periods = ? WHERE id = ?',
                (str(last_upd), meta_id))
    conn.commit()


def query_splitter(filters, id, nowrite=False):
    """Splits a query into multiple queries if the original one is too big.

    Parameters:
    filters (pandas.core.frame.DataFrame): A dataframe with FedStat filters.
    id (str): id of a FedStat indicator.
    nowrite (bool): If False, the data is written into the database.

    Returns:
    parsed (pandas.core.frame.DataFrame): A dataframe with parsed data from FedStat.
    Directly returns this dataframe when nowrite=True.

    Knowledge from Coursera courses:
    Course 1 Programming for Everybody for declaring a function,
    using 'for' loops and conditional 'if' statements.
    Course 2 Python Data Structures for working with different data structures
    (strings and lists).
    Course 3 Using Python to Access Web Data for parts where data gets retreived
    from FedStat website.
    """
    chunk_size = 1000000
    overall_df = pd.DataFrame()
    if query_size(filters) > chunk_size:
        # split queries by year
        years = filters[filters["filter_id"] == "3"]["value_id"].values
        periods = filters[filters["filter_id"] == "33560"]["value_id"].values
        for i in years:
            subset = filters.drop(filters[(filters["value_id"] != i) & (filters["filter_id"] == "3")].index)
            split_further = query_size(subset) > chunk_size
            if split_further:
                # split queries by periods (months or quarters)
                for j in periods:
                    subset2 = subset.drop(subset[(subset["value_id"] != j) & (subset["filter_id"] == "33560")].index)
                    query = make_query(subset2)
                    # send request
                    counter = len(periods) * list(years).index(i) + list(periods).index(j) + 1
                    print('Downloading data, chunk ' + str(counter) + ' out of ' + str(len(periods) * len(years)))
                    request = Request('https://fedstat.ru/indicator/data.do?format=sdmx',
                                      urlencode(query, doseq=True).encode())
                    response = urlopen(request, timeout=300)
                    # EXCEPTION CATCHER NECESSARY
                    print('Data retrieved.')
                    parse_sdmx(response, id, nowrite=nowrite)
                    time.sleep(1)
                    # write to DB
            else:
                query = make_query(subset)
                # send request
                counter = list(years).index(i) + 1
                print('Downloading data, chunk ' + str(counter) + ' out of ' + str(len(years)))
                request = Request('https://fedstat.ru/indicator/data.do?format=sdmx',
                                  urlencode(query, doseq=True).encode())
                try:
                    response = urlopen(request, timeout=300)
                    print('Data retrieved.')
                    parse_sdmx(response, id, nowrite=nowrite)
                    time.sleep(1)
                except:
                    print('Failed to retrieve data from FedStat. \
                    Please try later')
                    exit()
    else:
        query = make_query(filters)
        # send request
        print('Retrieving data...')
        request = Request('https://fedstat.ru/indicator/data.do?format=sdmx',
                          urlencode(query, doseq=True).encode())
        try:
            response = urlopen(request, timeout=300)
            print('Data retrieved.')
        except:
            print('Failed to retrieve data from FedStat. Please try later')
            exit()
        if nowrite:
            parsed = parse_sdmx(response, id, nowrite=nowrite)
            return parsed
        else:
            parse_sdmx(response, id, nowrite=nowrite)


def load_data(id):
    """Loads data from the database. This function is called when there is
    no new data on FedStat server for that indicator that is not already in
    the database.

    Parameters:
    id (str): id of a FedStat indicator.

    Returns:
    [result, titles] (list): A list whose elements are a dataframe result
    containing data for that indicator and a list of filters' titles.

    Knowledge from Coursera courses:
    Course 1 Programming for Everybody for declaring a function,
    using 'for' loops and conditional 'if' statements.
    Course 2 Python Data Structures for working with different data structures
    (strings and lists).
    Course 4 Using Databases with Python for retreiving data from the database.
    """
    cur.execute('SELECT columns FROM Metadata WHERE dataset=?', (id, ))
    item = cur.fetchone()
    for row in item:
        colnames = row.split(';')
    colnames = colnames[:-1]
    beg = 'SELECT '
    mid = 'FROM Data' + str(id)
    end = ' ON '
    # for some filter fields it can be more convenient for the user
    # to have codes instead of text values
    # so we add these fields to the output
    add = ['s_OKPD2', 's_OKPD', 's_OKATO']
    for col in colnames:
        try:
            cur.execute('SELECT max(id) FROM ' + col)
            row = cur.fetchone()
        except:
            row = None
        if row is not None:
            beg = beg + col + '.fed_title, '
            if col in add:
                beg = beg + col + '.fed_id, '
            mid = mid + ' JOIN ' + col
            end = end + 'Data' + str(id) + '.' + col + ' = ' + col + '.id ' + 'and '
        else:
            beg = beg + 'Data' + str(id) + '.' + col + ', '

    script = beg.rstrip(', ') + ' ' + mid + end[:-4] + ' ORDER BY Data' + str(id) + '.TIME'
    cur.execute(script)
    # rename fields with codes so that there are no duplicates
    for i in add:
        if i in colnames:
            colnames.insert(colnames.index(i)+1, i+'_id')
    # load data into dataframe
    result = pd.DataFrame(cur, columns=colnames)
    # now we select human-readable titles for filter values
    titles = list()
    for i in colnames:
        script = 'SELECT field_title FROM Filternames WHERE field_id = ?'
        cur.execute(script, (i, ))
        try:
            row = cur.fetchone()[0]
        except:
            row = None
        if row is None:
            titles.append(i)
        else:
            titles.append(row)
    return [result, titles]


def get_data(id, force_upd=False):
    """Loads data needed to make reports.

    Parameters:
    id (str): id of a FedStat indicator.
    force_upd (bool): If True, data in the database gets updated.

    Results in either the data being written into the database by calling
    query_splitter() that in turn calls parse_sdmx() that calls write_db(),
    when there is no data on the selected indicator in the database,
    or the data being uploaded from the database.

    Knowledge from Coursera courses:
    Course 1 Programming for Everybody for declaring a function,
    using 'for' loops and conditional 'if' statements.
    Course 2 Python Data Structures for working with different data structures
    (strings, lists and dictionaries).
    Course 4 Using Databases with Python for retreiving data from the database.
    """
    filters = get_structure(id, force_upd=force_upd)
    if filters.empty:
        print('Error in getting the internal Fedstat filter structure. \
        Please, try again.')
        return []
    try:
        cur.execute('SELECT max(id) FROM Data' + str(id))
        row = cur.fetchone()
    except:
        row = None
    if row is not None and not force_upd:
        # get last updated date on server
        upd = get_periods(id)
        # get last updated date locally
        cur.execute('SELECT periods FROM Metadata WHERE dataset = ?',
                    (str(id),))
        row = cur.fetchone()
        for item in row:
            last_upd = eval(item)
        # check which dates are present on server and are not downloaded
        missing = [x for x in upd if x not in last_upd]
        if len(missing) != 0:
            # get period ids dictionary to map value to ids
            period_ids = dict(filters.loc[filters["filter_id"] == '33560',
                                          ["value_id",
                                           "value_title"]].values.tolist())
            period_ids = dict((v, k) for k, v in period_ids.items())
            # get unique missing filter values
            missing = list(missing)
            missing_years = list({[i[0] for i in missing]})
            missing_periods = list({[i[1] for i in missing]})
            templist = list()
            for _, missing_year in enumerate(missing_years):
                templist.append(['3', 'Год', str(missing_year),
                                 str(missing_year), 'columnObjectIds'])
            for _, missing_period in enumerate(missing_periods):
                templist.append(['33560', 'Период',
                                 period_ids[missing_period],
                                 missing_period, 'columnObjectIds'])
            temp = pd.DataFrame(templist, columns=filters.columns)
            # change the filters to load missing dates
            filters_augm = pd.concat([filters.loc[(filters["filter_id"] != "3") & (filters["filter_id"] != "33560")], temp])
            query_splitter(filters_augm, id)
    elif force_upd and row is not None:
        cur.execute('DROP TABLE Data' + str(id))
        cur_str.execute('DELETE from Structure WHERE database=?', (str(id), ))
        query_splitter(filters, id)
    elif row is None:
        query_splitter(filters, id)
    print('Data processing successful.')


def get_periods(id):
    """Fetches from FedStat server what periods are available for an indicator.
    The function is needed to compare dates that are on the server
    and those that are already in the database.

    Parameters:
    id (str): id of a FedStat indicator.

    Returns:
    set(periods): a set of elements from the list of periods.
    A set is used so that there are no duplicates in values.

    Knowledge from Coursera courses:
    Course 1 Programming for Everybody for declaring a function,
    using a 'for' loop.
    Course 2 Python Data Structures for working with a list.
    """
    filters = get_structure(id)
    filters_short = pd.concat(
        [filters.loc[(filters["filter_id"] != "3") & (filters["filter_id"] != "33560") & (filters["filter_id"] != "57956")].groupby('filter_id').first().reset_index(),
         filters.loc[(filters["filter_id"] == "3") | (filters["filter_id"] == "33560") | (filters["filter_id"] == "57956")]])
    result = query_splitter(filters_short, id, nowrite=True)
    periods = list()
    result["TIME"] = result.TIME.astype(int)
    for r in result[["TIME", "PERIOD"]].itertuples(index=False):
        periods.append(r)
    return set(periods)


def monetary_value():
    # 34118 - area introduced
    get_data('34118')
    [area, area_col] = load_data('34118')
    s_mosh = ['Жилые здания многоквартирные',]

    # 31452 - average price
    get_data('31452')
    [price, price_col] = load_data('31452')
    s_vidryn     = ['Первичный рынок жилья',]
    s_OKATO      = ['Центральный федеральный округ', 'Северо-Западный федеральный округ',
                    'Южный федеральный округ (с 29.07.2016)', 'Северо-Кавказский федеральный округ',
                    'Приволжский федеральный округ', 'Уральский федеральный округ',
                    'Сибирский федеральный округ', 'Дальневосточный федеральный округ']
    S_TIPKVARTIR = ['Все типы квартир',]
    PERIOD       = ['I квартал', 'II квартал', 'III квартал', 'IV квартал']
    years        = [2019, 2020, 2021]
    # filtering of Prices
    price = price[price['s_vidryn'].isin(s_vidryn)]
    price = price[price['TIME'].isin(years)]
    price = price[price['PERIOD'].isin(PERIOD)]
    price = price[price['S_TIPKVARTIR'].isin(S_TIPKVARTIR)]
    price = price[price['s_OKATO'].isin(s_OKATO)]

    # Filtering of Areas
    area = area.set_index(['s_OKATO', 'TIME', 'PERIOD'])
    area = area[area['s_mosh'].isin(s_mosh)]

    # Calculate values for the 1st quarter of 2019
    quarter_index = pd.MultiIndex.from_arrays([s_OKATO + s_OKATO, [2019, 2019, 2019, 2019, 2019, 2019, 2019, 2019,
                                                                   2020, 2020, 2020, 2020, 2020, 2020, 2020, 2020],
                                               ['I квартал', 'I квартал', 'I квартал', 'I квартал',
                                                'I квартал', 'I квартал', 'I квартал', 'I квартал',
                                                'I квартал', 'I квартал', 'I квартал', 'I квартал',
                                                'I квартал', 'I квартал', 'I квартал', 'I квартал']])
    first_quarter = pd.DataFrame(index=quarter_index, columns=['VALUE', ])
    first_quarter = first_quarter.sort_index()
    for district in s_OKATO:
        for year in [2019, 2020]:
            temp_quarter = area.loc[(district, year, ['январь', 'февраль', 'март']), 'VALUE']
            first_quarter.loc[(district, year, 'I квартал'), 'VALUE'] = temp_quarter.sum()

    area = pd.concat([area, first_quarter])
    area = area.loc[(s_OKATO, years, PERIOD)]
    area = area.reset_index()
    price_area = price.merge(area, left_on=['s_OKATO', 'TIME', 'PERIOD'], right_on=['s_OKATO', 'TIME', 'PERIOD'])

    # Calculate monetary value
    monetary = pd.Series([], dtype='float64')
    for i in range(len(price_area)):
        monetary[i] = price_area['VALUE_x'][i] * price_area['VALUE_y'][i] / 1000
    monetary = monetary.round(2)
    price_area.insert(5, "Monetary Value in RUB, millions", monetary)

    # clean up the data
    price_area = price_area.drop(columns='s_OKATO_id_y')
    price_area['s_mosh'] = price_area['s_mosh'].fillna(value='Жилые здания многоквартирные')
    price_area['EI_y'] = price_area['EI_y'].fillna(value='тысяча квадратных метров общей площади')

    # associate technical column names with human-readable column names
    col_names = ['TIME', 'PERIOD', 's_OKATO', 's_OKATO_id_x', 's_vidryn', 's_mosh', 'EI_x', 'EI_y', 'VALUE_x',
                 'VALUE_y',
                 'S_TIPKVARTIR']
    nice_names = ['Year', 'Period', 'Federal District', 'Federal District (id)', 'Type of Market', 'Type of Building',
                  'Unit of Price', 'Unit of Area', 'Average Price', 'Area Introduced', 'Type of Flats']
    name_dict = dict(zip(col_names, nice_names))

    # Replace technical names with human-readable
    price_area = price_area.rename(columns=name_dict)
    area = area.rename(columns=name_dict)
    price = price.rename(columns=name_dict)

    # Create pivot-tables
    price_area_pivot = price_area.pivot(index='Federal District', columns=['Year', 'Period'],
                                  values="Monetary Value in RUB, millions")
    price_pivot = price.pivot(index='Federal District', columns=['Year', 'Period'], values='VALUE')
    area_pivot = area.pivot(index='Federal District', columns=['Year', 'Period'], values='VALUE')

    price_area_pivot.to_csv('Monetary Value Report.csv', encoding='utf-8')
    print('\n', price_area_pivot, '\n')

    inp_d = {'both': [price_pivot, area_pivot], 'pr': [price_pivot, ], 'ar': [area_pivot, ], 'no': []}
    while True:
        inp = input('Would you like to add separate tables for price and area? pr/ar/both/no ')
        if inp in inp_d.keys():
            break
        else:
            print('Please enter the correct option')
    price_pivot.name = 'price'
    area_pivot.name = 'area'
    for i in inp_d[inp]:
        i.to_csv(i.name + '.csv', encoding='utf-8')
        print('\n', i, '\n')


def monthly_introduction():
    get_data('34118')
    [area, titles_area] = load_data('34118')
    PERIOD = ['январь', 'февраль', 'март', 'апрель', 'май', 'июнь',
              'июль', 'август', 'сентябрь', 'октябрь', 'ноябрь', 'декабрь']
    area = area[area['PERIOD'].isin(PERIOD)]

    # categorize months for the sort
    area['PERIOD'] = pd.Categorical(area['PERIOD'], categories=PERIOD, ordered=True)
    area.sort_values(['s_OKATO', 'TIME', 'PERIOD'])

    # associate technical names with human-readable
    col_names = ['TIME', 'PERIOD', 's_OKATO', 's_OKATO_id_x', 's_mosh', 'EI', 'VALUE']
    nice_names = ['Year', 'Period', 'Federal District', 'Federal District (id)', 'Type of Building',
                  'Unit of Area', 'Area Introduced']
    name_dict = dict(zip(col_names, nice_names))

    # Replace technical names with human-readable
    area = area.rename(columns=name_dict)

    area_pivot = area.pivot(index=['Federal District', 'Type of Building'],
                            columns=['Year', 'Period'], values='Area Introduced')
    area_pivot.to_csv('Area Introduced Monthly.csv', encoding='utf-8')

    print(area_pivot)


# def monthly_prices():

def user_interface():
    """Asks the user to enter id of a FedStat indicator to make reports on.
    Then proceeds to fetch the data from FedStat and writes it into the database.
    If there is no data on FedStat website that is not already in the database,
    data gets loaded from the database right away.
    If the user presses Enter, asks for a new indicator id.

    If the user presses Ctrl+C, stops.

    Knowledge from Coursera courses:
    Course 1 Programming for Everybody for declaring a function,
    using 'for' loops and conditional 'if' statements.
    Course 2 Python Data Structures for working with strings.
    """
    print('Press Ctrl+C to exit')
    a = ''
    while a == '':
        try:
            print('''Which report would you like to get?  
            1. Monetary Value of New Flats  
            2. Monthly Introduction of New Living Space''')
            rep = int(input('Enter the number of report: '))
            if rep == 1:
                monetary_value()
            elif rep == 2:
                monthly_introduction()
            elif rep == 3:

            print('Press Ctrl+C to exit. Press Enter to make a new report')
            a = str(input())
        except KeyboardInterrupt:
            exit()

user_interface()
