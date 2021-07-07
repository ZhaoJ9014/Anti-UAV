import cv2
import numpy

video_id = 5
#cap = cv2.VideoCapture('Users/yhmac/Desktop/演示视频/{}/{}.mp4'.format(vedio_id,vedio_id))
cap = cv2.VideoCapture('/home/dell/metric_uav/metric_tracker/crops_raw/{}.avi'.format(video_id))
anno_file = open('/home/dell/metric_uav/metric_tracker/crop_gt/new_video_{}_0.txt'.format(video_id))
annotations = anno_file.readlines()
annotations = [list(map(float,annotation.split())) for annotation in annotations]

anno_file_test = open('/home/dell/metric_uav/metric_tracker/pytracking/video_{}_0.txt'.format(video_id))
annotations_test = anno_file_test.readlines()
annotations_test = [list(map(int,annotation.split())) for annotation in annotations_test]
#print(len(annotations)）
    #print(annotations)


g_i = -1
cv2.namedWindow('imshow')
while(g_i<len(annotations)-1):
# Capture frame-by-frames
    ret, frame = cap.read()
    g_i += 1
    position = annotations[g_i]
    tposition = annotations_test[g_i]
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)#画面颜色
    #cv2.rectangle(frame, (int(round(position[0])), int(round(position[1]))), (int(round(position[0]+position[2])), int(round(position[1]+position[3]))), (0,0,255), 2)
    cv2.rectangle(frame, (tposition[0], tposition[1]), (tposition[0]+tposition[2], tposition[1]+tposition[3]), (0,255,0), 2)
    
    cv2.imshow('imshow',frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break
# When everything done, release the capture


#new_file = open('/Users/lyhmac/Desktop/default/new_video_ir{}.txt".format(vedio_id)', 'w')
#new.file.writelines(annotations)
#new_file.close()
anno_file.close()
cap.release()
cv2.destroyAllWindows()
