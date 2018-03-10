# The different locations of the electrodes correspond to different position of electrodes on the heart;
# the data received was organized by each "channel"
channels = ["DI", "DII", "DIII", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]

# the first batch of measurements are the widths of the wave and other extraneous statistics
partOfWave = ["width Q", "width R", "width S", "width R prime", "width S prime", "Num. of intrinsic deflections",
              "Ragged R wave", "Diphasic Derivation of R wave",
              "Ragged P wave", "Diphasic Derivation of P wave",
              "Ragged T wave", "Diphasic Derivation of T wave"]

# the 2nd set of measurements are the amplitudes of the certain parts of the wave & some averages (look directly below)
amplitude = ["amp JJ", "amp Q", "amp R", "amp S", "amp R prime", "amp S prime", "amp P", "amp T", "val QRSA", "val QRSTA"]

# the beginning attributes are these; never repeated
onceStr = "Age,Sex,Height,Weight,QRS duration,P-R interval,Q-T interval,T interval,P interval,QRS,T,P,QRST,J,Heart rate,"

# the final string placeholder
finStr = ""

# for each channel, we need to add every part of wave
for e in channels:
    for i in partOfWave:
        finStr = finStr + e + " " + i + ","
        # DI: width Q,

# for each channel, add all amplitudes
for e in channels:
    for i in amplitude:
        finStr = finStr + e + " " + i + ","
        # DI: ampJJ


# the final string has to have the beginning header then the
# compounded finished string (without ending commas and extra space) and finally the identifier label
finStr = onceStr + finStr[:-2] + ",Identifier"
print(finStr)

# save the string to a text file
text_file = open("Output.txt", "w")
text_file.write(finStr)
text_file.close()
