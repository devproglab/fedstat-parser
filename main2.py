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
conn = sqlite3.connect('data.sqlite')
cur = conn.cursor()
conn_str = sqlite3.connect('structure.sqlite')
cur_str = conn_str.cursor()

def get_structure(id, force_upd=False):
    cur_str.execute('''CREATE TABLE IF NOT EXISTS Structure
        (id INTEGER PRIMARY KEY, database INTEGER, filter_id INTEGER,
         value_id INTEGER, filter_type INTEGER)''')
    cur_str.execute('''CREATE TABLE IF NOT EXISTS Filters
        (filter_id INTEGER PRIMARY KEY, filter_title TEXT)''')
    cur_str.execute('''CREATE TABLE IF NOT EXISTS Filter_values
        (value_id INTEGER PRIMARY KEY, value_title TEXT)''')
    cur_str.execute('''CREATE TABLE IF NOT EXISTS Filter_types
        (filter_type INTEGER PRIMARY KEY AUTOINCREMENT, filter_type_title TEXT UNIQUE)''')
    #try to get structure
    cur_str.execute('SELECT id FROM Structure WHERE database=? LIMIT 1', (str(id), ))
    row = cur_str.fetchone()
    if row is not None and not force_upd:
        print('Reading data structure...')
        clmns = ['filter_id', 'fiter_title', 'value_id', 'value_title', 'filter_type']
        cur_str.execute('''SELECT Structure.filter_id, Filters.filter_title, Structure.value_id, Filter_values.value_title, 
        Filter_types.filter_type_title 
        FROM Structure 
        JOIN Filters on Structure.filter_id=Filters.filter_id
        JOIN Filter_values on Structure.value_id=Filter_values.value_id
        JOIN Filter_types on Structure.filter_type=Filter_types.filter_type WHERE database = ?
        ''', (id, ))
        filters = pd.DataFrame(cur_str, columns = clmns)
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
        else:
            data = response.read().decode()
            print('Processing data structure...')
            # extract the necessary part from raw javascript and modify it to be proper json
            pattern = re.compile(r"title:(.+?)\}\);", re.DOTALL | re.MULTILINE)
            results = pattern.findall(data)[0].strip()
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
            filters = pd.DataFrame(columns=('filter_id', 'filter_title', 'value_id', 'value_title'))
            for i in range(0, len(df2)):
                # extract titles of filters' values
                titles = re.findall("title': '(.+?)'", str(df2[i][1]["values"]))
                # extract internally used ids of filters' values
                keys = re.findall("'([0-9]+?)': {'title'", str(df2[i][1]["values"]))
                # merge into df with filter ids and filter titles
                titles = pd.DataFrame(
                    {"value_title": titles, "value_id": keys, "filter_title": len(titles) * [df2[i][1]["title"]],
                     "filter_id": len(titles) * [df2[i][0]]})
                filters = pd.concat([filters, titles])
            # replace "root" value title with dataset name
            filters.loc[filters["filter_id"] == "0", "value_title"] = results["title"]
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
            layout = pd.DataFrame({"filter_id": fields, "filter_type": nfields})
            if not any(layout["filter_id"] == "0"):
                layout = pd.concat([layout, pd.DataFrame({'filter_id': '0',
                                                          'filter_type': 'filterObjectIds'}, index=[0])])
            layout["filter_id"] = layout["filter_id"].astype(str)
            filters = pd.merge(filters, layout, how='left')
            # REMOVE LATER: WRITE STRUCTURE TO FILE
            filters['filter_id'] = filters.filter_id.astype(str)
            for index, row in filters.iterrows():
                cur_str.execute('''INSERT OR IGNORE INTO Filters (filter_id, filter_title) VALUES (?, ?)''', (str(row['filter_id']), row['filter_title']))
                cur_str.execute('''INSERT OR IGNORE INTO Filter_values (value_id, value_title) VALUES (?, ?)''', (str(row['value_id']), row['value_title']))
                cur_str.execute('''INSERT OR IGNORE INTO Filter_types (filter_type_title) VALUES (?)''', (str(row['filter_type']),))
                cur_str.execute('SELECT filter_type FROM Filter_types WHERE filter_type_title = ? ', (str(row['filter_type']), ))
                filter_type = cur_str.fetchone()[0]
                cur_str.execute('''INSERT INTO Structure (filter_id,
                  value_id, filter_type, database) VALUES (?, ?, ?, ?)''', (str(row['filter_id']), row['value_id'], filter_type, id ) )
            conn_str.commit()
            print('Data structure retrieved.')
    return filters


def query_size(filterdata):
    S = filterdata['filter_id'].value_counts().values.prod()
    return S


def make_query(filterdata):
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

def index_appender(list):
    newlist = []
    for i, v in enumerate(list):
        totalcount = list.count(v)
        count = list[:i].count(v)
        newlist.append(v + str(count + 1) if totalcount > 1 else v)
    return newlist

def parse_sdmx(response, id, force_upd = False, nowrite=False):
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
    for i in range(0, len(fields_id)):
        loc = 'CodeLists/structure:CodeList/[@id="' + fields_id[i] + '"]/structure:Code'
        temp = list()
        for node in tree.findall(loc, ns):
            temp.append(node.attrib["value"])
        fields_codes.append(temp)
    # get filter value names
    fields_values = list()
    for i in range(0, len(fields_id)):
        loc = 'CodeLists/structure:CodeList/[@id="' + fields_id[i] + '"]/structure:Code/structure:Description'
        temp = list()
        for node in tree.findall(loc, ns):
            temp.append(node.text)
        fields_values.append(temp)
    # rename possible duplicates caused by internal rosstat code changes for Sibersky & Dalnevostochny Federal Districts
    idx = fields_id.index('s_OKATO')
    fields_codes[idx] = [w.replace('035', '041') for w in fields_codes[idx]]
    fields_codes[idx] = [w.replace('036', '042') for w in fields_codes[idx]]
    # extract data from the xml structure
    # begin with table creation
    script = 'CREATE TABLE IF NOT EXISTS Data' + str(id) + ' (id INTEGER PRIMARY KEY AUTOINCREMENT UNIQUE, Time INTEGER, Value REAL)'
    cur.executescript(script)
    order = list()
    cur.execute('SELECT max(id) FROM Data' + str(id))
    try:
        r = cur.fetchone()[0]
    except:
        r = None
    for node in tree.findall('DataSet/generic:Series[generic:SeriesKey]', ns)[0]:
        for child in node.findall('generic:Value/[@concept]', ns):
            feature = child.attrib["concept"]
            script = 'CREATE TABLE IF NOT EXISTS ' + str(
                feature) + ' (id INTEGER PRIMARY KEY AUTOINCREMENT UNIQUE, fed_id TEXT UNIQUE, fed_title TEXT UNIQUE)'
            cur.executescript(script)
            order.append(feature)
            # need to check if table in the DB already exists
            if r is not None:
                a = 0
            else:
                script = 'ALTER TABLE Data' + str(id) + ' ADD ' + str(feature)
                cur.execute(script)
    conn.commit()
    #now we know names of columns
    colnames = order + ["TIME", "VALUE"]
    #send all filter data to DB
    for i in range(0, len(fields_id)):
        c = fields_id[i] #save filtername we are working with
        for j in range(0, len(fields_values[i])): #iterate trough all values for each filter
            script = 'INSERT OR IGNORE INTO ' + str(c) + ' (fed_id, fed_title) VALUES ( ?, ? )' #add filter decoders into DB
            cur.execute(script, (fields_codes[i][j], str(fields_values[i][j])) )
    # save filter data for EI and PERIOD
    for node in tree.findall('DataSet/generic:Series/generic:Attributes', ns):
        for child in node.findall('generic:Value/[@concept]', ns):
            feature = child.attrib["concept"]
            value =  child.attrib["value"]
            script = 'INSERT OR IGNORE INTO ' + str(
                feature) + ' (fed_title) VALUES (?)'
            cur.execute(script, (value, ))
    conn.commit()

    for node in tree.findall('DataSet/generic:Series', ns):
        temp_ds = []
        count = 0
        for child in node.findall('generic:SeriesKey//', ns):
            script = 'SELECT id FROM ' + fields_id[count] + ' WHERE fed_id=?' #build table relations
            v = child.attrib["value"]
            if v == '035':
                v = '041'
            if v == '036':
                v = '042'
            cur.execute(script, (v, ))
            for item in cur:
                temp_ds.append(item[0])
            count += 1
        for child in node.findall('generic:Attributes//', ns):
            script = 'SELECT id FROM ' + child.attrib['concept'] + ' WHERE fed_title=?' #build table relations
            cur.execute(script, (child.attrib["value"], ))
            for item in cur:
                temp_ds.append(item[0])
        for child in node.findall('generic:Obs/generic:Time', ns):
            temp_ds.append(int(child.text))
        for child in node.findall('generic:Obs/generic:ObsValue', ns):
            value = re.sub(',', '.', child.attrib["value"])
            temp_ds.append(float(value))

        #Create SQLite command for any number of columns
        script = 'INSERT INTO Data' + str(id) + ' ('
        left = ')  VALUES ('
        for col in colnames:
            script = script + col + ' , '
            left = left + '?,'
        script = script.rstrip(' , ')
        left = left.rstrip(',') + ')'
        script = script + left
        RowToUpload = tuple(temp_ds)
        cur.execute(script, RowToUpload)

    conn.commit()
    #write all column names into a string
    cols = ''
    for item in colnames:
        cols = cols + item + ';'
    cur.execute('CREATE TABLE IF NOT EXISTS Metadata (id INTEGER PRIMARY KEY AUTOINCREMENT UNIQUE, dataset INTEGER UNIQUE, columns TEXT, periods TEXT)')
    #if metadata for this dataset already exists, do nothing, else append new row
    # cur.execute('SELECT max(TIME) FROM Data' + str(id))
    # row = cur.fetchone()
    # for item in row:
    #     t = item
    cur.execute('INSERT OR IGNORE INTO Metadata (dataset, columns) VALUES (?, ?)', (id, cols))
    conn.commit()
    parsed = load_data(id)
    last_upd = set(list(tuple(x) for x in parsed.loc[:, ["TIME", "PERIOD"]].values))
    cur.execute('SELECT id FROM Metadata WHERE dataset = ?', (str(id), ))
    row = cur.fetchone()
    for item in row:
        meta_id = item
    cur.execute('UPDATE Metadata SET periods = ? WHERE id = ?', (str(last_upd), meta_id))
    conn.commit()
    return parsed


def query_splitter(filters, id, nowrite=False):
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
                    chunk_parsed = parse_sdmx(response, id, nowrite=nowrite)
                    overall_df = pd.concat([overall_df, chunk_parsed])
                    time.sleep(1.5)
                    # write to DB
            else:
                query = make_query(subset)
                # send request
                counter = list(years).index(i) + 1
                print('Downloading data, chunk ' + str(counter) + ' out of ' + str(len(years)))
                request = Request('https://fedstat.ru/indicator/data.do?format=sdmx',
                                  urlencode(query, doseq=True).encode())
                response = urlopen(request, timeout=300)
                # EXCEPTION CATCHER NECESSARY
                print('Data retrieved.')
                chunk_parsed = parse_sdmx(response, id, nowrite=nowrite)
                #overall_df = pd.concat([overall_df, chunk_parsed])
                time.sleep(1.5)
                # WRITER HERE
    else:
        #print(filters)
        query = make_query(filters)
        # send request
        print('Retrieving data...')
        request = Request('https://fedstat.ru/indicator/data.do?format=sdmx',
                          urlencode(query, doseq=True).encode())
        response = urlopen(request, timeout=300)
        # EXCEPTION CATCHER NECESSARY
        print('Data retrieved.')
        chunk_parsed = parse_sdmx(response, id, nowrite=nowrite)
        overall_df = pd.concat([overall_df, chunk_parsed])
    return overall_df

def load_data(id):
    cur.execute('SELECT columns FROM Metadata WHERE dataset=?', (id, ))
    item = cur.fetchone()
    for row in item:
        colnames = row.split(';')
    colnames = colnames[:-1]
    beg = 'SELECT '
    mid = 'FROM Data' + str(id)
    end = ' ON '
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

    for i in add:
        if i in colnames:
            colnames.insert(colnames.index(i)+1, i+'_id')
    result = pd.DataFrame(cur, columns = colnames)
    return result

def get_data(id, force_upd=False):
    filters = get_structure(id, force_upd=force_upd)
    if filters.empty:
        print('Error in getting the internal Fedstat filter structure. Please, try again.')
        return []
    try:
        cur.execute('SELECT max(id) FROM Data' + str(id))
        row = cur.fetchone()
    except:
        row = None
    if row is not None and not force_upd:
        #overall_df = load_data(id)
        #print('Data retrieved.')
        # get last updated date on server
        upd = get_periods(id)
        # get last updated date locally
        cur.execute('SELECT periods FROM Metadata WHERE dataset = ?', (str(id),))
        row = cur.fetchone()
        for item in row:
            last_upd = eval(item)
        #last_upd = set(list(tuple(x) for x in overall_df.loc[:, ["TIME", "PERIOD"]].values))
        # check which dates are present on server and are not downloaded
        missing = [x for x in upd if x not in last_upd]
        if len(missing) == 0:
            return load_data(id)
        else:
            # get period ids dictionary to map value to ids
            period_ids = dict(filters.loc[filters["filter_id"] == '33560', ["value_id", "value_title"]].values.tolist())
            period_ids = dict((v, k) for k, v in period_ids.items())
            # get unique missing filter values
            missing = list(missing)
            missing_years = list(set([i[0] for i in missing]))
            missing_periods = list(set([i[1] for i in missing]))
            templist = list()
            for i in range(0, len(missing_years)):
                templist.append(['3', 'Год', str(missing_years[i]), str(missing_years[i]), 'columnObjectIds'])
            for i in range(0, len(missing_periods)):
                templist.append(['33560', 'Период', period_ids[missing_periods[i]], missing_periods[i], 'columnObjectIds'])
            temp = pd.DataFrame(templist, columns=filters.columns)
            # change the filters to load missing dates
            filters_augm = pd.concat([filters.loc[(filters["filter_id"] != "3") & (filters["filter_id"] != "33560")], temp])
            result = query_splitter(filters_augm, id)
            #result = pd.concat([overall_df, result])
            #result = result.drop_duplicates()
    elif force_upd and row is not None:
        cur.execute('DROP TABLE Data' + str(id))
        cur_str.execute('DELETE from Structure WHERE database=?', (str(id), ) )
        result = query_splitter(filters, id)
    elif row is None:
        result = query_splitter(filters, id)
    print('Data processing successful.')
    print(result.head(2))


def get_periods(id):
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



print('Which data do you want to work with?')
print('34118 - housing data, 58971 - food, 31452 - more housing data')
id = str(input())
parsed = get_data(id) #31074 s bad
#parsed = get_data("58971", force_upd=True)
#parsed = get_periods(id)
