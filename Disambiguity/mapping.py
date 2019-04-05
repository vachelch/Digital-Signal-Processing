import re
import sys

From = sys.argv[1]
To = sys.argv[2]

ZhuYin_Big5 = {}

# Read inverted Big5-ZhuYin.map
with open(From, 'r', encoding = 'big5', errors='ignore') as f:
	for i, row in enumerate(f.readlines()):
		row = re.sub('[\s+]', '', row)

		Big5 = row[0]
		ZhuYin = set([word[0] for word in row[1:].split('/')])

		for zy in ZhuYin:
			if zy not in ZhuYin_Big5:
				ZhuYin_Big5[zy] = [Big5]
			else:
				ZhuYin_Big5[zy].append(Big5)


# Write ZhuYin_Big5 to map
with open(To, 'w', encoding = 'big5') as g:
	for zy, big5s in ZhuYin_Big5.items():
		g.write('{} {}\n'.format(zy, ' '.join(big5s)))








