import csv

inpath = 'holdout_key.csv'
outpath = 'out_holdout_key.csv'

with open(outpath, 'wb') as f_out:
    writer = csv.writer(f_out)
    with open(inpath, 'rb') as f_in:
        reader = csv.reader(f_in)
        reader.next()
        writer.writerow(['clip', 'preictal'])
        for row in reader:
            clip = row[0]
            clip = clip.replace('_test', '')
            label = row[1]
            writer.writerow([clip, label])
