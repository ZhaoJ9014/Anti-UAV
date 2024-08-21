import cv2

video_path= r"D:\MOT20-01-raw.mp4"
out_path= r"E:\Images"


is_all_frame=True
sta_frame=1
end_frame=1000

time_interval=8

def saveimage(image,num):
    address=out_path+"\\"+str(num)+".jpg"
    print("save success")
    cv2.imwrite(address,image)


cap=cv2.VideoCapture(video_path)
success,frame=cap.read()

i=0
j=0

while success:
    i+=1
    if i%time_interval==0:
        if not is_all_frame:
            if sta_frame <=i <=end_frame:
                j += 1
                print("save frame",j)
                saveimage(frame,j)
            else:
                break
        else:
            j+=1
            print("save frame",j)
            saveimage(frame, j)
    success, frame = cap.read()

cap.release()