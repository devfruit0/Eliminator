import time
import numpy as np
import cv2
import os
import gpiozero

DETECTION = "person"
SAVE = True
RUN_TIME_SEC = 60
SHOOT_TOLERANCE = 7
PIXELS_PER_METER = 125
DISTANCE_TO_OBJECT = 5
MIN_AREA = 100
MAX_AREA = 1000
CONFIDENCE = 0.01
BORDER = 8
FRAMES_TO_ENSURE = 1
FRAMES_TO_CANCEL_SHOOTING = 8
CORRECT_THRESHOLD_PROPORTION = 0
CHANGE_THRESHOLD = 0.2

# All Pins are BCM
RELAY_PIN = 17
SERVO_PIN = 18
RELAY = gpiozero.OutputDevice(RELAY_PIN)
SERVO = gpiozero.AngularServo(SERVO_PIN, min_angle=-45, max_angle=45)
SERVO.mid()

# TODO: find optimal values
LOWER_SQUIRREL_HSV = np.array([20, 0, 50])
UPPER_SQUIRREL_HSV = np.array([130, 255, 180])
LABELS = open(os.path.abspath("yolo-coco/coco.names")).read().strip().split("\n")

np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

weightsPath = os.path.sep.join(["yolo-coco", "yolov3-tiny.weights"])
configPath = os.path.sep.join(["yolo-coco", "yolov3-tiny.cfg"])

print("Loading YOLO v3 Tiny from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
layer_names = net.getLayerNames()
layer_names = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
WIDTH = len(frame[0])
HEIGHT = len(frame)

if SAVE:
    out = cv2.VideoWriter('output_videos/output' + str(int(time.time())) + '.mkv',
                          cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 1, (WIDTH, HEIGHT))

print("Width:", WIDTH, "Height:", HEIGHT)

key = 0
last_frame = None
checking_times = 0
tracker = None
correct_detections = 0
# current_angle = 0
start_time = time.time()


def getAngle(box_x, box_width):
    box_center = int(box_x) + int(box_width / 2)
    distance_to_center = abs(int(WIDTH / 2) - box_center)
    print("Distance to Center in px:", distance_to_center)

    # Trying to use more one liners bc garbage code makes me laugh XD
    calc_angle = abs(int(np.arctan((distance_to_center / PIXELS_PER_METER) / DISTANCE_TO_OBJECT) * 57.2958))

    # left side of the image is positive
    if box_center < WIDTH / 2:
        return -calc_angle
    # right side of the image is negative
    else:
        return calc_angle


def fire():
    print("***FIRING***")
    RELAY.on()
    time.sleep(2)
    RELAY.off()
    time.sleep(1)
    SERVO.angle = 0
    time.sleep(1)


def adjustAngle(a):
    print("Moving to:", a)
    SERVO.angle = a
    time.sleep(1)


# partially code from pyImageSearch: https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
def non_max_suppression_fast(boxes, overlap_thresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
    # initialize the list of picked indexes
    for i in range(len(boxes)):
        boxes[i] = np.array((boxes[i][0], boxes[i][1], boxes[i][0] + boxes[i][2], boxes[i][1] + boxes[i][3]))

    pick = []
    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlap_thresh)[0])))

    fixed_boxes = boxes[pick].astype("int")
    final_boxes = []
    for i in range(len(fixed_boxes)):
        final_boxes.append((fixed_boxes[i][0], fixed_boxes[i][1],
                            fixed_boxes[i][2] - fixed_boxes[i][0],
                            fixed_boxes[i][3] - fixed_boxes[i][1]))
    # return only the bounding boxes that were picked using the
    # integer data type
    return final_boxes


def getContours(_last_frame, _gray):
    frameDelta = cv2.absdiff(_last_frame, _gray)
    thresh = cv2.threshold(frameDelta, 50, 255, cv2.THRESH_BINARY)[1]

    if np.sum(thresh) > (WIDTH * HEIGHT * CHANGE_THRESHOLD):
        return []

    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=1)
    cont, _ = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return cont


def make_detection_from_region(_region):
    blob = cv2.dnn.blobFromImage(_region, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(layer_names)

    confidences = []
    classIDs = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            classIDs.append(classID)
            confidences.append(confidence)

    # Trying to make the code as cancer as possible
    return LABELS[int(classIDs[confidences.index(max(confidences))])]


# return max probability detection for each class from the image
def make_detections_from_image(_image):
    blob = cv2.dnn.blobFromImage(_image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(layer_names)

    confidences = []
    classIDs = []
    boxes = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # classIDs.append(classID)
            # confidences.append(confidence)
            if confidence > CONFIDENCE:
                box = detection[0:4] * np.array([WIDTH, HEIGHT, WIDTH, HEIGHT])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top
                # and and left corner of the bounding box
                _x = int(centerX - (width / 2))
                _y = int(centerY - (height / 2))

                # update our list of bounding box coordinates,
                # confidences, and class IDs
                boxes.append([_x, _y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # a list that contains all of the classIDs filtered through
    classes_sorted = []
    confidences_sorted = []
    boxes_sorted = []

    for i in range(len(classIDs)):
        found = False
        for j in range(len(classes_sorted)):
            if classIDs[i] in classes_sorted[j]:
                classes_sorted[j].append(classIDs[i])
                confidences_sorted[j].append(confidences[i])
                boxes_sorted[j].append(boxes[i])
                found = True
        if not found:
            classes_sorted.append([classIDs[i]])
            confidences_sorted.append([confidences[i]])
            boxes_sorted.append([boxes[i]])

    max_confidences_ids = [confidences_id_set.index(max(confidences_id_set))
                           for confidences_id_set in confidences_sorted]

    # Trying to make the code as cancer as possible
    return [LABELS[classes_sorted[i][max_confidences_ids[i]]] for i in range(len(max_confidences_ids))], \
           [boxes_sorted[i][max_confidences_ids[i]] for i in range(len(max_confidences_ids))]


while True:
    if SAVE and time.time() - start_time > RUN_TIME_SEC:
        break

    ret, frame = cap.read()
    targeting_frame = frame.copy()

    if checking_times == 0:
        print("==SEEKING==")
        if DETECTION == "squirrel":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
            if last_frame is not None:
                potential_objects = []
                objects = []

                contours = getContours(last_frame, gray)
                for c in contours:
                    (x, y, w, h) = cv2.boundingRect(c)

                    # Break the loop if the aspect ratio isn't within a range
                    # or if the area of the contour is within a range
                    if cv2.contourArea(c) < MIN_AREA or cv2.contourArea(c) > MAX_AREA or w / h < 1 / 3 or w / h > 3:
                        continue

                    cv2.rectangle(frame, (np.clip(x - BORDER, 0, WIDTH), np.clip(y - BORDER, 0, HEIGHT)),
                                  (np.clip(x + w + BORDER, 0, WIDTH), np.clip(y + h + BORDER, 0, HEIGHT)),
                                  (0, 255, 0), 2)

                    potential_objects.append([x, y, w, h])

                potential_suppressed_objects = non_max_suppression_fast(np.array(potential_objects), 0.2)

                for (x, y, w, h) in potential_suppressed_objects:
                    print((x, y, w, h))
                    region = targeting_frame[np.clip(y - BORDER, 0, HEIGHT):np.clip(y + h + BORDER, 0, HEIGHT),
                                             np.clip(x - BORDER, 0, WIDTH):np.clip(x + w + BORDER, 0, WIDTH)]

                    regional_detection = make_detection_from_region(region)

                    if not regional_detection == "bird" or regional_detection is None:
                        objects.append((x, y, w, h))

                if len(objects) > 0:
                    print(DETECTION, "FOUND")
                    detected_object = objects[0]

                    for _object in objects:
                        if _object[0] < detected_object[0]:
                            detected_object = _object

                    tracker = cv2.TrackerCSRT_create()
                    tracker.init(targeting_frame, detected_object)

                    checking_times += 1

            last_frame = gray

        # if trying to detect something else without the motion tracking part
        else:
            detections, regions = make_detections_from_image(targeting_frame)
            print(detections)
            detected_object = None
            if DETECTION in detections:
                detected_object = regions[detections.index(DETECTION)]

            if detected_object is not None:
                print(DETECTION, "FOUND")
                tracker = cv2.TrackerCSRT_create()
                tracker.init(targeting_frame, tuple(detected_object))

                cv2.rectangle(frame, (detected_object[0], detected_object[1]),
                              (detected_object[0] + detected_object[2], detected_object[1] + detected_object[3]),
                              (45, 90, 255), 3)

                checking_times += 1

    # Enter Validation mode
    elif checking_times < FRAMES_TO_ENSURE:
        print("==VALIDATION==")
        checking_times += 1

        (success, box) = tracker.update(targeting_frame)
        box = (np.clip(int(box[0]), 0, WIDTH), np.clip(int(box[1]), 0, HEIGHT),
               np.clip(int(box[0] + box[2]), 0, WIDTH) - int(box[0]),
               np.clip(int(box[1] + box[3]), 0, HEIGHT) - int(box[1]))

        # np.clip(
        if success:
            print(box)
            region = targeting_frame[box[1]:box[1] + box[3], box[0]:box[0] + box[2]]

            regional_detection = make_detection_from_region(region)
            print("Detection: ", regional_detection)
            if DETECTION == "squirrel":
                # More awesome one liners!!!!!
                if cv2.inRange(cv2.cvtColor(targeting_frame, cv2.COLOR_BGR2HSV), LOWER_SQUIRREL_HSV,
                               UPPER_SQUIRREL_HSV)[box[1] + int(box[3]/2), box[0] + int(box[2]/2)] == 1:
                    correct_detections += 1
                if regional_detection == "bird":
                    checking_times = 0

            elif DETECTION == regional_detection:
                correct_detections += 1
        else:
            checking_times = 0

        # Final iteration
        if checking_times == FRAMES_TO_ENSURE:
            correct_detections = 0
            if correct_detections < CORRECT_THRESHOLD_PROPORTION * FRAMES_TO_ENSURE:
                checking_times = 0

    else:
        print("==TARGETING==")
        checking_times += 1
        (success, box) = tracker.update(targeting_frame)
        box = tuple(map(int, box))
        if success:
            angle = getAngle(box[0], box[2])
            print(box)

            cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[0]) + int(box[2]),
                                                                        int(box[1]) + int(box[3])), (225, 0, 0), 3)

            cv2.putText(frame, 'Detected A: ' + str(angle) + ', Servo A: ' + str(SERVO.angle) +
                        ', Moving A: ' + str(45 * ((-1) ** checking_times)) + ', T: ' +
                        str(int(time.time() - start_time)), (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            print("Detected Angle:", angle)
            print("Servo Angle:", SERVO.angle)

            if abs(angle) < SHOOT_TOLERANCE:
                fire()
                checking_times = 0
                # current_angle = 0
            else:
                adjustAngle(45 * ((-1) ** checking_times))
                # current_angle = angle + current_angle
        else:
            checking_times = 0
            # current_angle = 0
            adjustAngle(0)
        if checking_times - FRAMES_TO_ENSURE > FRAMES_TO_CANCEL_SHOOTING:
            checking_times = 0
            # current_angle = 0
            adjustAngle(0)

    if SAVE:
        out.write(frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord(' '):
        key = cv2.waitKey()
    if key == ord('q') or not ret:
        break

print("wrapping up...")

if SAVE:
    out.release()

cap.release()
cv2.destroyAllWindows()
