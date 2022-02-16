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
except:
    print(
        'Error: some of the required packages are missing. Please install the dependencies by running pip install -r '
        'requirements.txt')
    exit()


def get_structure(id, force_upd=False):
    # REPLACE TO DB QUERY
    if os.path.exists(str('struct/' + id + '.csv')) and not force_upd:
        print('Reading data structure...')
        filters = pd.read_csv(str('struct/' + id + '.csv'), sep=';')
        filters['filter_id'] = filters.filter_id.astype(str)
        print('Data structure retrieved.')
    else:
        # load all possible filter values and codes from the fedstat page
        print('Downloading data structure...')
        url = "https://fedstat.ru/indicator/" + id
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
            results = json.loads(results)
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
            filters.to_csv(str('struct/' + id + '.csv'), sep=';', encoding="utf-8-sig", index=False)
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
    meta = filterdata.loc[filterdata["filter_id"] == "0"].values[0]
    query = [('id', meta[2]),
             ('title', meta[3])] + filterdata.drop_duplicates(
        subset=["filter_id"]).loc[:, ['filter_type', 'filter_id']].to_records(index=None).tolist() \
            + list(zip([['selectedFilterIds'] * len(p)][0], p))
    # format query to json-styled request accepted by the server
    query_json = {}
    for i in query:
        query_json.setdefault(i[0], []).append(i[1])
    return query_json


def parse_sdmx(response):
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
    # extract data from the xml structure
    datalist = list()
    for node in tree.findall('DataSet/generic:Series', ns):
        temp = list()
        for child in node.findall('generic:SeriesKey//', ns):
            temp.append(child.attrib["value"])
        for child in node.findall('generic:Attributes//', ns):
            temp.append(child.attrib["value"])
        for child in node.findall('generic:Obs/generic:Time', ns):
            temp.append(child.text)
        for child in node.findall('generic:Obs/generic:ObsValue', ns):
            temp.append(child.attrib["value"])
        datalist.append(temp)
    # sometimes the order of filters does not correspond to the description - fix it
    order = list()
    for node in tree.findall('DataSet/generic:Series[generic:SeriesKey]', ns)[0]:
        for child in node.findall('generic:Value/[@concept]', ns):
            order.append(child.attrib["concept"])
    # merge all data into dataframe
    colnames = order + ["TIME", "VALUE"]
    parsed = pd.DataFrame(datalist, columns=colnames)
    # create decoding dictionaries to map internal codes to human-readable names
    decoders = list()
    for i in range(0, len(fields_id)):
        tempdf = pd.DataFrame({fields_id[i]: fields_codes[i], str(fields_id[i] + '_title'): fields_values[i]})
        decoders.append(tempdf)
    # append values from dictionaries to the dataframe
    for i in range(0, len(fields_id)):
        parsed = pd.merge(parsed, decoders[i], how='left')
    return parsed


def query_splitter(filters):
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
                    chunk_parsed = parse_sdmx(response)
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
                chunk_parsed = parse_sdmx(response)
                overall_df = pd.concat([overall_df, chunk_parsed])
                time.sleep(1.5)
                # WRITER HERE
                print('Write to DB')
    else:
        query = make_query(filters)
        # send request
        print('Retrieving data...')
        request = Request('https://fedstat.ru/indicator/data.do?format=sdmx',
                          urlencode(query, doseq=True).encode())
        response = urlopen(request, timeout=300)
        # EXCEPTION CATCHER NECESSARY
        print('Data retrieved.')
        chunk_parsed = parse_sdmx(response)
        overall_df = pd.concat([overall_df, chunk_parsed])
    return overall_df


def get_data(id, force_upd=False):
    filters = get_structure(id, force_upd=force_upd)
    # CHANGE TO DB
    if os.path.exists(str('data/' + id + '.csv')) and not force_upd:
        overall_df = pd.read_csv(str('data/' + id + '.csv'), sep=';')
        overall_df = overall_df.dropna()
        overall_df["TIME"] = overall_df.TIME.astype(int)
        print('Data retrieved.')
        # get last updated date on server
        upd = get_periods(id)
        # get last updated date locally CHANGE TO DB
        last_upd = set(list(tuple(x) for x in overall_df.loc[:, ["TIME", "PERIOD"]].values))
        # check which dates are present on server and are not downloaded
        missing = [x for x in upd if x not in last_upd]
        if len(missing) == 0:
            return overall_df
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
            result = query_splitter(filters_augm)
            result = pd.concat([overall_df, result])
            result = result.drop_duplicates()
            result.to_csv(str('data/' + id + '.csv'), sep=';',
                          encoding="utf-8-sig", index=False)
    else:
        result = query_splitter(filters)
        result.to_csv(str('data/' + id + '.csv'), sep=';',
                      encoding="utf-8-sig", index=False)
    print('Data processing successful.')
    return result


def get_periods(id):
    filters = get_structure(id)
    filters_short = pd.concat(
        [filters.loc[(filters["filter_id"] != "3") & (filters["filter_id"] != "33560") & (filters["filter_id"] != "57956")].groupby('filter_id').first().reset_index(),
         filters.loc[(filters["filter_id"] == "3") | (filters["filter_id"] == "33560") | (filters["filter_id"] == "57956")]])
    result = query_splitter(filters_short)
    periods = list()
    result["TIME"] = result.TIME.astype(int)
    for r in result[["TIME", "PERIOD"]].itertuples(index=False):
        periods.append(r)
    return set(periods)


parsed = get_data("34118")
# parsed = get_data("58971", force_upd=True)
# = get_periods('31074')