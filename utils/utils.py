import csv

def convert_int(x):
    try:
        return int(x)
    except ValueError:
        return x

def read_csv(file_name):
    output_list = []
    with open(file_name, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)
        for row in csvreader:
            _ = [convert_int(x) for x in row]
            output_list.append(_)
    return output_list
