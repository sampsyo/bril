import csv
import sys


def normalize():
    # Read input CSV.
    reader = csv.DictReader(sys.stdin)
    in_data = list(reader)

    # Get normalization baselines.
    baselines = {
        row['benchmark']: int(row['result'])
        for row in in_data
        if row['run'] == 'baseline'
    }

    # Write output CSV back out.
    writer = csv.DictWriter(sys.stdout, reader.fieldnames)
    writer.writeheader()
    for row in in_data:
        row['result'] = int(row['result']) / baselines[row['benchmark']]
        writer.writerow(row)


if __name__ == '__main__':
    normalize()
