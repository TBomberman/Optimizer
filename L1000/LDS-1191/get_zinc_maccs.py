import csv
import urllib.request
import time
from threading import Timer

# go through the entire zinc csv file
# hit each link

zinc_path = "/data/datasets/gwoo/zinc15/"
zinc_urls_file = "ZINC-downloader-2D-smi.uri"

def find_nth(haystack, needle, n):
    start = haystack.find(needle)
    while start >= 0 and n > 1:
        start = haystack.find(needle, start+len(needle))
        n -= 1
    return start

def download_zinc():
    with open(zinc_path + zinc_urls_file, "r") as csv_file:
        reader = csv.reader(csv_file, dialect='excel', delimiter=' ')
        for url in reader:
            # "http://files.docking.org/2D/HJ/HJAC.smi"
            list = url[0].split('/')
            a = list[3]
            b = list[4]
            c = list[5]
            save_file_name = a + "_" + b + '_' + c
            try:
                u = urllib.request.urlopen(url[0])
            except:
                print('could not open url', url[0])
                continue
            f = open(zinc_path + save_file_name, 'wb')
            meta = u.info()
            file_size = int(meta.get("Content-Length"))
            print ("Downloading: %s Bytes: %s" % (zinc_path + save_file_name, file_size))

            file_size_dl = 0
            block_sz = 8192
            while True:
                buffer = u.read(block_sz)
                if not buffer:
                    break
                file_size_dl += len(buffer)
                f.write(buffer)
                status = r"%10d  [%3.2f%%]" % (file_size_dl, file_size_dl * 100. / file_size)
                # status = status + chr(8) * (len(status) + 1)
                print(status)
                time.sleep(60)
            f.close()

def check_number_entries():
    import os

    zinc_path = "/data/datasets/gwoo/zinc/"
    count = 0
    for root, dirs, files in os.walk(zinc_path):
        for filename in files:
            extension = filename.split('.')
            if not extension[1].startswith('tab'):
                continue
        # zinc_urls_file = "ZincCompounds_InStock_maccs.tab"
            zinc_urls_file = filename
            print(zinc_urls_file)
            with open(zinc_path + zinc_urls_file, "r") as csv_file:
                reader = csv.reader(csv_file, dialect='excel', delimiter=' ')

                for row in reader:
                    count += 1
                    if count % 500000 == 0:
                        print(str(count), row[0])

check_number_entries()
