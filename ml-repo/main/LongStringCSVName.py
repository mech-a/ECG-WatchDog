import numpy

channels = ["DI", "DII", "DIII", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]

partOfWave = ["width Q", "width R", "width S", "width R prime", "width S prime", "Num. of intrinsic deflections",
              "Ragged R wave", "Diphasic Derivation of R wave",
              "Ragged P wave", "Diphasic Derivation of P wave",
              "Ragged T wave", "Diphasic Derivation of T wave"]

amplitude = ["amp JJ", "amp Q", "amp R", "amp S", "amp R prime", "amp S prime", "amp P", "amp T", "val QRSA", "val QRSTA"]

onceStr = "Age,Sex,Height,Weight,QRS duration,P-R interval,Q-T interval,T interval,P interval,QRS,T,P,QRST,J,Heart rate,"

finStr = ""

for e in channels:
    for i in partOfWave:
        finStr = finStr + e + " " + i + ","

for e in channels:
    for i in amplitude:
        finStr = finStr + e + " " + i + ","

finStr = onceStr + finStr[:-2] + ",Identifier"
print(finStr)


text_file = open("Output.txt", "w")
text_file.write(finStr)
text_file.close()
