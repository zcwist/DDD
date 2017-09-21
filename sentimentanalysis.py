import sys  

reload(sys)  
sys.setdefaultencoding('utf8')

import nltk
from nltk import tokenize

filename = "dataset/interview/H2Woah.txt"
f = open(filename)
html = open('demo.html','a')
html.write("""<!DOCTYPE html>
<meta charset="UTF-8">
<html>
<head>
<title>Page Title</title>
</head>
<body>""")

data = f.readlines()
for line in data:

	sentences = tokenize.sent_tokenize(line)

	from nltk.sentiment.vader import SentimentIntensityAnalyzer
	# nltk.download("vader_lexicon")

	sid = SentimentIntensityAnalyzer()
	html.write("<p>")

	for sentence in sentences:
		score = sid.polarity_scores(sentence)["compound"]
		if score > 0:
			html.write("<font color='red'>")
		elif score < 0:
			html.write("<font color='green'>")
		else:
			html.write("<font color='black'>")
		html.write(sentence)
		html.write("</font>")
		
	html.write("</p>")
	html.write("</body")
	