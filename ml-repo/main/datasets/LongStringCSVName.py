import numpy

channels = ["DI", "DII", "AVR", "AFL", "V1", "V2", "V3", "V4", "V5", "V6"]

partOfWave = ["Q", "R", "S", "R prime", "S prime"]

amplitude = ["JJ", "Q", "R", "S", "R prime", "S prime", "P ", "T", "QRSA", "QRSTA"]

finStr = ""

for e in channels:
    for i in partOfWave:
        finStr = finStr + e + ": width " + i + ", "

for e in channels:
    for i in amplitude:
        finStr = finStr + e + ": amp " + i + ", "

print(finStr[:-1])

text_file = open("Output.txt", "w")
text_file.write("Purchase Amount: %s" % finStr[:-2])
text_file.close()
