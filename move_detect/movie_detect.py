import cv2

filepath = "vtest.avi"
cap = cv2.VideoCapture(filepath)

while True:
    # 1フレームずつ取得する。
    ret, frame = cap.read()
    if not ret:
        break  # フレームの取得に失敗または動画の末尾に到達した場合

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(30)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()