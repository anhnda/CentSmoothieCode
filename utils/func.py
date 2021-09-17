def add_dict_counter(d,e,c=1):
    try:
        d[e] += c
    except:
        d[e] = c
