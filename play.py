import tensorflow as tf
import numpy as np

def convertToReadable(act):
	ret = ""
	i = act % 10
	if (act - i) == 0:
		ret += "A"
	elif (act - i) == 10:
		ret += "B"
	elif (act - i) == 20:
		ret += "C"
	elif (act - i) == 30:
		ret += "D"
	elif (act - i) == 40:
		ret += "E"
	elif (act - i) == 50:
		ret += "F"
	elif (act - i) == 60:
		ret += "G"
	elif (act - i) == 70:
		ret += "H"
	elif (act - i) == 80:
		ret += "I"
	elif (act - i) == 90:
		ret += "A"
	ret += str(i + 1)


battleModel = tf.keras.models.load_model('saved_model/my_model')

print(battleModel.summary())

# 0=hidden/empy -1=Miss 1=Hit 11=Sunk Patrol 22=Sunk Sub 33=Sunk Crusier 44=Sunk Battleship 55= Sunk Carrier
row0 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
row1 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
row2 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
row3 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
row4 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
row5 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
row6 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
row7 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
row8 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
row9 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

obs = row1 + row2 + row3 + row4 + row5 + row6 + row7 + row8 + row9 + row0

sobs = [obs]
print(obs.__len__())

prediction = battleModel.predict(sobs)[0]

action = np.argmax(prediction)

print("Reccomended shot is space #" + str(action))