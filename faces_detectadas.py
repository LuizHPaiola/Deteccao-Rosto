import cv2

detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
imagem = cv2.imread("imagens/ryan4.jpg")

imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
faces_detectadas = detector.detectMultiScale(imagemCinza, scaleFactor = 1.07)
print(faces_detectadas)

for x,y,l,a in faces_detectadas:
    imagem = cv2.rectangle(imagem, (x,y), (x+l, y+a), (0,0,255), 2)

cv2.imshow("Imagem da face", imagem)
cv2.waitKey()
cv2.destroyWindow("Imagem da face")