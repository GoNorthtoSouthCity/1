import cv2
import numpy as np
from collections import deque
#import serial

cnt = 0
bef_p = res_p = [[0, 0] for i in range(9)]
stop_flag = 0
def mmain(image):
    # # 调用函数并传入图像路径2
    # 测试图片
    global cnt
    global bef_p
    global stop_flag
    global turning_point_color
    global wall
    wall = [
        [[0, 1, 0, 1], [0, 0, 1, 1], [1, 0, 1, 1], [0, 1, 0, 1], [0, 0, 1, 1], [0, 0, 0, 1], [1, 0, 1, 1], [0, 1, 0, 1],
         [1, 0, 0, 1], [1, 1, 0, 0]],
        [[0, 1, 0, 0], [1, 0, 1, 1], [0, 1, 0, 1], [1, 0, 1, 0], [1, 1, 0, 1], [1, 1, 0, 0], [0, 1, 0, 1], [1, 0, 1, 0],
         [0, 1, 1, 0], [1, 0, 1, 0]],
        [[1, 1, 0, 0], [0, 1, 0, 1], [0, 0, 0, 0], [0, 0, 1, 1], [1, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0], [0, 1, 1, 1],
         [0, 0, 1, 1], [1, 0, 0, 1]],
        [[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 1, 0], [0, 0, 0, 1], [0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 1, 0], [0, 0, 0, 1],
         [1, 0, 0, 1], [1, 1, 0, 0]],
        [[0, 1, 1, 0], [0, 0, 0, 1], [1, 0, 0, 1], [1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 0, 1], [1, 1, 0, 0],
         [0, 1, 0, 0], [1, 0, 1, 0]],
        [[0, 1, 0, 1], [1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 0, 0], [0, 1, 1, 0],
         [0, 0, 1, 0], [1, 0, 0, 1]],
        [[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 0], [1, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 1, 0], [1, 0, 0, 1],
         [0, 1, 0, 1], [1, 0, 0, 0]],
        [[0, 1, 1, 0], [0, 0, 1, 1], [1, 0, 1, 1], [0, 1, 0, 0], [1, 0, 0, 0], [0, 1, 0, 1], [0, 0, 1, 1], [0, 0, 0, 0],
         [1, 0, 1, 0], [1, 1, 0, 0]],
        [[0, 1, 0, 1], [1, 0, 0, 1], [0, 1, 0, 1], [1, 0, 1, 0], [1, 1, 0, 0], [1, 1, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0],
         [0, 1, 1, 1], [1, 0, 0, 0]],
        [[1, 1, 1, 0], [0, 1, 1, 0], [1, 0, 1, 0], [0, 1, 1, 1], [0, 0, 1, 0], [0, 0, 1, 1], [1, 0, 1, 0], [0, 1, 1, 1],
         [0, 0, 1, 1], [1, 0, 1, 0]]]
    # 设置起点终点 左下角（0,9） 右上角（9，0）
    begin_point = [0, 9]
    end_point = [9, 0]
    res_path = [[]]
    # 透视变换前的坐标，程序自动识别获取
    p1 = [150, 150]
    p2 = [150, 150]
    p3 = [150, 150]
    p4 = [150, 150]
    dis1 = 10000
    dis2 = 10000
    dis3 = 10000
    dis4 = 10000

    # 透视变换后的坐标
    p11 = [0, 0]
    p22 = [300, 0]
    p33 = [0, 300]
    p44 = [300, 300]

    points = [[]]
    res_points = [[]]

    rows, cols = 10, 10#十行十列
    #wall = [[[0, 0, 0, 0] for _ in range(cols)] for _ in range(rows)]
    # 下上右左
    # output = [[(0, 0) for _ in range(10)] for _ in range(10)]
    turning_count = 0
    pos_x = [int(48 + 22.7 * i) for i in range(10)]
    pos_y = [int(48 + 22.7 * i) for i in range(10)]

    res_p = [[0, 0] for i in range(9)]#9个目标点
    res_p_visited = [0 for i in range(9)]
    res_p[8] = [end_point[0], end_point[1]]#出口
    res_p_visited[8] = 1

    def get_wall(img):
        _gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#颜色转换

        # 二值化处理,将图像中的像素值转换为或255，从而将图像转换为黑白二值图像
        ret, thresh = cv2.threshold(_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        for row in range(10):
            for col in range(10):
                for i in range(9, 15):
                    if (thresh[pos_y[col] + i][pos_x[row]] >= 200):
                        wall[row][col][0] = 1
                        # cv2.circle(warped_image,(pos_x[row],pos_y[col]+10),2,(0,100,255))
                    if (thresh[pos_y[col] - i][pos_x[row]] >= 200):
                        wall[row][col][1] = 1
                        # cv2.circle(warped_image,(pos_x[row],pos_y[col]-10),2,(0,100,255))
                    if (thresh[pos_y[col]][pos_x[row] + i] >= 200):
                        wall[row][col][2] = 1
                        # cv2.circle(warped_image,(pos_x[row]+10,pos_y[col]),2,(0,100,255))
                    if (thresh[pos_y[col]][pos_x[row] - i] >= 200):
                        wall[row][col][3] = 1
                        # cv2.circle(warped_image,(pos_x[row]-10,pos_y[col]),2,(0,100,255))

        return thresh

#铺平宝藏图
    def inverse_perspective_transform(image):
        # 定义原图中四个角点的坐标
        source_points = np.float32([p1, p2, p3, p4])

        # 定义变换后的图像中对应的四个角点坐标
        destination_points = np.float32([p11, p22, p33, p44])

        # 计算透视变换矩阵
        transform_matrix = cv2.getPerspectiveTransform(source_points, destination_points)

        # 进行逆透视变换
        warped_image = cv2.warpPerspective(image, transform_matrix, (image.shape[1], image.shape[0]))

        return warped_image

    # def mouse_callback(event, x, y, flags, param):
    #     if event == cv2.EVENT_LBUTTONDOWN:
    #         print("Clicked coordinates: ({}, {})".format(x, y))

    def bfs(start, rows, cols):#输入现在的的坐标，十行,十列
        visited = [[False] * cols for _ in range(rows)]  # 用于跟踪已访问的元素
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # 下、上、you、zuo

        # 从(0, 0)开始进行BFS
        queue = deque([start])#双端队列
        visited[0][0] = False
        path = {}  # 记录节点的前驱节点
        flag = 0
        while queue:
            x, y = queue.popleft()#出队列
            # cv2.circle(warped_image, (int(48 + 22.7 * x), int(48 + 22.7 * y)), 3, (start[0]*30, 0, 255),-1)
            # cv2.imshow("t", warped_image)
            # cv2.waitKey(100)
            # 判断是否达到目标点
            if [x, y] in res_p and flag == 1:
                ff = 0
                for i in range(9):
                    if x == res_p[i][0] and y == res_p[i][1]:
                        if res_p_visited[i] == 1:
                            ff = 1
                        res_p_visited[i] = 1
                        break
                if ff == 0:
                    return get_path(start, (x, y), path)
            index = -1
            flag = 1
            # 处理当前节点的邻居节点
            for dx, dy in directions:
                index = index + 1
                if (wall[x][y][index] == 1):
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < rows and 0 <= ny < cols and not visited[nx][ny]:
                    queue.append((nx, ny))
                    visited[nx][ny] = True
                    path[(nx, ny)] = (x, y)  # 记录邻居节点的前驱节点

        return None  # 未找到目标点

    def get_path(start, end, _path):
        # 根据前驱节点回溯路径
        if end not in _path:
            return None

        current = end
        result = [current]

        while current != start:
            current = _path[current]
            result.append(current)

        result.reverse()
        return result

    def find_largest_contour(image):
        # Otsu's 二值化处理
        # 将图像转换为灰度图像
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 寻找轮廓
        _,contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 找到最大的连通域
        max_area = 0
        max_contour = None
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > max_area:
                max_area = area
                max_contour = contour

        # 绘制最大连通域的轮廓
        result = cv2.drawContours(image.copy(), [max_contour], -1, (0, 255, 0), 2)

        # 提取ROI
        x, y, w, h = cv2.boundingRect(max_contour)
        roi = image[y:y + h, x:x + w]

        return roi

    # 读取图像
    # image = cv2.resize(image, (300, 300))

    image = find_largest_contour(image)
    image = cv2.resize(image, (300, 300))
    # 二值化处理
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # # 膨胀操作
    # kernel = np.ones((7, 7), np.uint8)
    # dilated2 = cv2.erode(thresh, kernel, iterations=1)
    # 膨胀操作
    kernel = np.ones((2, 2), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=1)
    # 创建结构元素（这里使用矩形结构元素）
    #
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    # # 膨胀操作
    # thresh = cv2.dilate(thresh, kernel, iterations=1)
    # 连通域分析
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)
    if len(labels) < 12:
        return image
    # 获取连通域信息
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        left = stats[label, cv2.CC_STAT_LEFT]
        top = stats[label, cv2.CC_STAT_TOP]
        width = stats[label, cv2.CC_STAT_WIDTH]
        height = stats[label, cv2.CC_STAT_HEIGHT]
        if area / (width * height) >= 0.7 and width <= 11 and height <= 11:
            x = left + width / 2
            y = top + height / 2
            points.append([x, y])
            if pow(x, 2) + pow(y, 2) < dis1:
                dis1 = pow(x, 2) + pow(y, 2)
                p1 = [x, y]
            if pow(x - 300, 2) + pow(y, 2) < dis2:
                dis2 = pow(x - 300, 2) + pow(y, 2)
                p2 = [x, y]

            if pow(x, 2) + pow(y - 300, 2) < dis3:
                dis3 = pow(x, 2) + pow(y - 300, 2)
                p3 = [x, y]

            if pow(x - 300, 2) + pow(y - 300, 2) < dis4:
                dis4 = pow(x - 300, 2) + pow(y - 300, 2)
                p4 = [x, y]

            # 绘制连通域的外接矩形
            cv2.rectangle(image, (left, top), (left + width, top + height), (0, 255, 0), 2)

    warped_image = inverse_perspective_transform(image)
    img2 = warped_image
    # find_quadrilateral(image)
    tt = get_wall(warped_image)

    # 定义原图中四个角点的坐标
    source_points = np.float32([p1, p2, p3, p4])

    # 定义变换后的图像中对应的四个角点坐标
    destination_points = np.float32([p11, p22, p33, p44])

    # 计算透视变换矩阵
    transform_matrix = cv2.getPerspectiveTransform(source_points, destination_points)

    for p in points[1:]:
        origin_point = np.array([[p[0], p[1]], [p[0], p[1]], [p[0], p[1]], [p[0], p[1]]], dtype=np.float32).reshape(-1,
                                                                                                                    1,
                                                                                                                    2)
        t_point = cv2.perspectiveTransform(origin_point, transform_matrix)
        x = t_point[0][0][0]
        y = t_point[0][0][1]
        if x > 6 and y > 6 and x < 290 and y < 290:
            res_points.append(t_point[0])
            cv2.circle(warped_image, (int(t_point[0][0][0]), int(t_point[0][0][1])), 4, (0, 0, 255))

    # for i in range(10):
    #     for j in range(10):
    #         cv2.circle(warped_image, (int(48 + 22.7 * i), int(48 + 22.7 * j)), 7, (255, 0, 255))
    #         if wall[i][j][0] == 1:
    #             cv2.circle(warped_image, (int(48 + 22.7 * i), int(48 + 22.7 * j + 10)), 2, (0, 0, 255))
    #         if wall[i][j][1] == 1:
    #             cv2.circle(warped_image, (int(48 + 22.7 * i), int(48 + 22.7 * j - 10)), 2, (0, 0, 255))
    #         if wall[i][j][2] == 1:
    #             cv2.circle(warped_image, (int(48 + 22.7 * i + 10), int(48 + 22.7 * j)), 2, (0, 0, 255))
    #         if wall[i][j][3] == 1:
    #             cv2.circle(warped_image, (int(48 + 22.7 * i - 10), int(48 + 22.7 * j)), 2, (0, 0, 255))
    if len(res_points)!=9:
        print("printing res_points_length")
        print(len(res_points))
        return image
    i = 0
    for p in res_points[1:]:
        rp = [int(p[0][0]), int(p[0][1])]
        if len(rp) <= 1:
            continue
        rp[0] = round((rp[0] - 48) / 22.7)
        rp[1] = round((rp[1] - 48) / 22.7)
        res_p[i] = [rp[0], rp[1]]
        i = i + 1

    now = (begin_point[0], begin_point[1])
    c_r = 0
    c_b = 0

    for j in range(10):
        path = bfs(now, 10, 10)
        if path == None:
            break
        res_path.append(path)
        for i in range(len(path) - 1):
            start = (pos_x[path[i][0]], pos_y[path[i][1]])
            end = (pos_x[path[i + 1][0]], pos_y[path[i + 1][1]])
            if (wall[path[i][0]][path[i][1]][0] + wall[path[i][0]][path[i][1]][1] + wall[path[i][0]][path[i][1]][2] +
                    wall[path[i][0]][path[i][1]][3] <= 1):
                # 通过矢量计算向左向右或向前
                vec_a = path[i][0] - path[i - 1][0]
                vec_b = path[i][1] - path[i - 1][1]
                vec_c = path[i + 1][0] - path[i][0]
                vec_d = path[i + 1][1] - path[i][1]
                if (vec_a * vec_d - vec_b * vec_c == -1):
                    turning_point_color = (255, 0, 0)  # 左转蓝色
                    print("左转", turning_count)
                    # output[j][turning_count]=(turning_count,-1)

                elif (vec_a * vec_d - vec_b * vec_c == 1):
                    turning_point_color = (0, 255, 0)  # 右转绿色
                    print("右转", turning_count)
                    # output[j][turning_count]=(turning_count,1)
                elif (vec_a * vec_d - vec_b * vec_c == 0):
                    turning_point_color = (0, 0, 255)  # 直走红色
                    print("直走", turning_count)
                    # output[j][turning_count]=(turning_count,2)
                else:
                    break
                turning_count += 1
                cv2.circle(warped_image, (pos_x[path[i][0]], pos_y[path[i][1]]), 5, turning_point_color, -1)  # 标记转弯点
            cv2.line(warped_image, start, end, (c_r, c_b, 0), thickness=2)

        c_r = j * 80
        c_b = 255 - j * 30
        now = path[-2]
        # now = path[-1]

    res_p_visited[8] = 0
    path = bfs(now, 10, 10)
    if path == None:
        return img2
    res_path.append(path)
    for i in range(len(path) - 1):
        start = (pos_x[path[i][0]], pos_y[path[i][1]])
        end = (pos_x[path[i + 1][0]], pos_y[path[i + 1][1]])
        cv2.line(warped_image, start, end, (0, 255, 255), thickness=2)
    # return thresh
    if len(res_path) != 10:
        return img2

    if bef_p == res_p:
        cnt = cnt + 1
    else:
        cnt = 0
    bef_p = res_p

    if cnt <= 3:
        return img2
    # 打印路径
    res_x = []
    res_y = []
    print("路径:")
    for pp in res_path:
        if len(pp) >= 1:
            for p in pp:
                #if (p == (0,9)or p == (9,9)) or wall[p[0]][p[1]] != [1,1,0,0] and wall[p[0]][p[1]] != [0,0,1,1]:
                if wall[p[0]][p[1]] != [1,1,0,0] and wall[p[0]][p[1]] != [0,0,1,1]:
                    #print(p[0], p[1])
                    #if len(res_x)==0 or (p[0]!=res_x[-1] or p[1]!=res_y[-1]):
                    res_x.append(p[0])
                    res_y.append(p[1])
                    # print(wall[p[0]][p[1]])
    print(res_x)
    print(res_y)
    #print(*res_x, *res_y, sep="")
    stop_flag = 1
    return warped_image


# video_capture = cv2.VideoCapture(0)
video_capture = cv2.VideoCapture(0)

while True:
    # 读取视频流的一帧
    ret, frame = video_capture.read()

    # 检查是否成功读取帧
    if not ret:
        break
    img = mmain(frame)
    # 在窗口中显示帧
    cv2.imshow('Video', img)
    cv2.imshow('bef', frame)
    cv2.waitKey(100)
    # 按下 'q' 键退出循环
    if stop_flag == 1:
        if cv2.waitKey(0) or 0xFF == ord('q'):
            break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放视频流对象和关闭窗口
video_capture.release()
cv2.destroyAllWindows()