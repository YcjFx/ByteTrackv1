import argparse
import os
import os.path as osp
import time
import cv2
import numpy as np
import torch

from loguru import logger

#调用其他文件的类
from yolox.data.data_augment import preproc
#调用检测模型
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess
#作图类
from yolox.utils.visualize import plot_tracking
from yolox.tracker.byte_tracker import BYTETracker
#使用追踪结果
from yolox.tracking_utils.timer import Timer

#画图函数
def draw_area(im0):
    # 画电子围栏
    # (0,720),(1200,720)
    cv2.line(im0, (0, 720), (1200, 720), (0, 255, 255), 10)

    # (0,720),(0,630)
    cv2.line(im0, (0, 720), (0, 630), (0, 255, 255), 10)
    # (0,630),(440,210)
    cv2.line(im0, (0, 630), (440, 210), (0, 255, 255), 10)
    # (440,210),(520,260)
    cv2.line(im0, (440, 210), (520, 260), (0, 255, 255), 10)
    # (520,260),(605,310)
    cv2.line(im0, (520, 260), (605, 310), (0, 255, 255), 10)
    # (605,310),(810,520)
    cv2.line(im0, (605, 310), (810, 520), (0, 255, 255), 10)
    # (810,520),(1200,660)
    cv2.line(im0, (810, 520), (1200, 660), (0, 255, 255), 10)
    # (1200,640),(1200,720)
    cv2.line(im0, (1200, 660), (1200, 720), (0, 255, 255), 10)

    # 画辅助线(480,250),(1200,640) Dy=60
    # 下
    # (520,260),(605,310)
    cv2.line(im0, (520, 200), (605, 250), (255, 0, 0), 3)
    # (605,310),(810,520)
    cv2.line(im0, (605, 250), (810, 460), (255, 0, 0), 3)
    # (810,520),(1200,660)
    cv2.line(im0, (810, 460), (1200, 600), (255, 0, 0), 3)

    # 上
    # (520,260),(605,310)
    cv2.line(im0, (520, 320), (605, 370), (255, 0, 0), 3)
    # (605,310),(810,520)
    cv2.line(im0, (605, 370), (810, 580), (255, 0, 0), 3)
    # (810,520),(1200,660)
    cv2.line(im0, (810, 580), (1200, 720), (255, 0, 0), 3)

    # 商场通道
    # (440,210),(605,120)
    cv2.line(im0, (440, 210), (605, 120), (0, 125, 255), 10)
    # (605,120),(810,120)
    cv2.line(im0, (605, 120), (810, 120), (0, 125, 255), 10)
    # (810,120),(1200,360)
    cv2.line(im0, (810, 120), (1200, 360), (0, 125, 255), 10)
    # (1200,360),(1200,640)
    cv2.line(im0, (1200, 360), (1200, 660), (0, 125, 255), 10)

    # 指定像素区域
    x, y, w, h = 10, 10, 250, 120
    # 画矩形并填充白色
    cv2.rectangle(im0, (x, y), (x + w, y + h), (255, 255, 255), -1)


#区域判断函数：
def judge_area(x1, y1, x2, y2):
    #x1, y1为检测框中心点
    point = (int((int(x1)+int(x2))/2), int(y2))

    #第一分段
    if (0 <= point[0] < 440):
        # (0, 630),(440, 210),
        k1 = (210 - 630) / (440 - 0)
        # 直线上的点,y_hat1商店边界
        y_hat1 = k1 * (point[0] - 0) + 630
        if (point[1] >= y_hat1):
            return 1   # 商店区域
        else:
            return 0

    #第二分段
    elif (440 <= point[0] < 520):
        # (440, 210),(520, 260),
        k1 = (260 - 210) / (520 - 440)
        # 直线上的点,y_hat1商店边界
        y_hat1 = k1 * (point[0] - 440) + 210
        y_hatup=y_hat1+60
        y_hatdown = y_hat1-60

        # (440, 210),(605, 120)
        k2 = (120 - 210) / (605 - 440)
        # 直线上的点，y_hat2通道边界
        y_hat2 = k2 * (point[0] - 440) + 210
        if (point[1] >= y_hatup):
            return 1 # 商店正区域
        elif (y_hat1 <= point[1] < y_hatup):
            return 11 # 商店边界区域
        elif (y_hatdown <= point[1] < y_hat1):
            return 22 # 过道靠商店边界区域
        elif (y_hat2 <= point[1] < y_hatdown):
            return 2  # 过道区域
        else:
            return 0

    #第三分段
    elif (520 <= point[0] < 605):
        # (520,260),(605,310),
        k1 = (310 - 260) / (605 - 520)
        # 直线上的点,y_hat1商店边界
        y_hat1 = k1 * (point[0] - 520) + 260
        y_hatup = y_hat1 + 60
        y_hatdown = y_hat1 - 60

        # (440,210),(605,120)
        k2 = (120 - 210) / (605 - 440)
        # 直线上的点，y_hat2通道边界
        y_hat2 = k2 * (point[0] - 440) + 210
        if (point[1] >= y_hatup):
            return 1   # 商店正区域
        elif (y_hat1 <= point[1] < y_hatup):
            return 11    # 商店边界区域
        elif (y_hatdown <= point[1] < y_hat1):
            return 22    # 过道靠商店边界区域
        elif (y_hat2 <= point[1] < y_hatdown):
            return 2    # 过道区域
        else:
            return 0

    # 第四分段
    elif (605 <= point[0] < 810):
        # (605,310),(810,520),
        k1 = (520 - 310) / (810 - 605)
        # 直线上的点,y_hat1商店边界
        y_hat1 = k1 * (point[0] - 605) + 310
        y_hatup = y_hat1 + 60
        y_hatdown = y_hat1 - 60

        # (605,120),(810, 120)
        k2 = (120 - 120) / (810 - 605)
        # 直线上的点，y_hat2通道边界
        y_hat2 = k2 * (point[0] - 605) + 120
        if (point[1] >= y_hatup):
            return 1  # 商店正区域
        elif (y_hat1 <= point[1] < y_hatup):
            return 11  # 商店边界区域
        elif (y_hatdown <= point[1] < y_hat1):
            return 22  # 过道靠商店边界区域
        elif (y_hat2 <= point[1] < y_hatdown):
            return 2  # 过道区域
        else:
            return 0

    # 第五分段
    elif (810 <= point[0] <= 1200):
        # (810,520),(1200,660)
        k1 = (660 - 520) / (1200 - 810)
        # 直线上的点,y_hat1商店边界
        y_hat1 = k1 * (point[0] - 810) + 520
        y_hatup = y_hat1 + 60
        y_hatdown = y_hat1 - 60

        # (810,120),(1200, 360)
        k2 = (360 - 120) / (1200 - 810)
        # 直线上的点，y_hat2通道边界
        y_hat2 = k2 * (point[0] - 810) + 120
        if (point[1] >= y_hatup):
            return 1  # 商店正区域
        elif (y_hat1 <= point[1] < y_hatup):
            return 11  # 商店边界区域
        elif (y_hatdown <= point[1] < y_hat1):
            return 22  # 过道靠商店边界区域
        elif (y_hat2 <= point[1] < y_hatdown):
            return 2  # 过道区域
        else:
            return 0

    #不检测区域
    else:
        return 0



IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]


def make_parser():
    parser = argparse.ArgumentParser("ByteTrack Demo!")
    #将image改为video
    parser.add_argument(
        "demo", default="video", help="demo type, eg. image, video and webcam"
    )
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")
    #默认测试数据路径
    parser.add_argument(
        #"--path", default="./datasets/mot/train/MOT17-05-FRCNN/img1", help="path to images or video"
        "--path", default="./videos/shop.mp4", help="path to images or video"
    )
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    #保存测试结果
    parser.add_argument(
        "--save_result",
        action="store_true",
        help="whether to save the inference result of image/video",
    )

    # exp file
    parser.add_argument(
        "-f",
        "--exp_file",
        # default=None,
        default="exps/example/mot/yolox_s_mix_det.py",
        type=str,
        help="pls input your expriment description file",
    )
    # parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    #默认权重
    parser.add_argument("-c", "--ckpt", default="models/bytetrack_s_mot17.pth.tar", type=str, help="ckpt for eval")
    #改为默认CPU
    parser.add_argument(
        "--device",
        # default="gpu",
        default="cpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument("--conf", default=None, type=float, help="test conf")
    parser.add_argument("--nms", default=None, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--fps", default=30, type=int, help="frame rate (fps)")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    # tracking args
    parser.add_argument("--track_thresh", type=float, default=0.5, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument(
        "--aspect_ratio_thresh", type=float, default=1.6,
        help="threshold for filtering out boxes of which aspect ratio are above the given value."
    )
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")
    return parser


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = osp.join(maindir, filename)
            ext = osp.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


def write_results(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores in results:
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1), h=round(h, 1), s=round(score, 2))
                f.write(line)
    logger.info('save results to {}'.format(filename))


class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        trt_file=None,
        decoder=None,
        device=torch.device("cpu"),
        fp16=False
    ):
        #模型
        self.model = model
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        if trt_file is not None:
            #torch2trt显卡使用库
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones((1, 3, exp.test_size[0], exp.test_size[1]), device=device)
            self.model(x)
            self.model = model_trt
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    #inference,推断函数
    def inference(self, img, timer):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = osp.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        img, ratio = preproc(img, self.test_size, self.rgb_means, self.std)
        img_info["ratio"] = ratio
        img = torch.from_numpy(img).unsqueeze(0).float().to(self.device)
        if self.fp16:
            img = img.half()  # to FP16

        with torch.no_grad():
            timer.tic()
            #调用模型,检测模型输出结果
            outputs = self.model(img)

            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre, self.nmsthre
            )

            #logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        return outputs, img_info


def image_demo(predictor, vis_folder, current_time, args):
    if osp.isdir(args.path):
        files = get_image_list(args.path)
    else:
        files = [args.path]
    files.sort()
    tracker = BYTETracker(args, frame_rate=args.fps)
    timer = Timer()
    results = []

    for frame_id, img_path in enumerate(files, 1):
        outputs, img_info = predictor.inference(img_path, timer)
        if outputs[0] is not None:
            online_targets = tracker.update(outputs[0], [img_info['height'], img_info['width']], exp.test_size)
            online_tlwhs = []
            online_ids = []
            online_scores = []
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(t.score)
                    # save results
                    results.append(
                        f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                    )
            timer.toc()
            online_im = plot_tracking(
                img_info['raw_img'], online_tlwhs, online_ids, frame_id=frame_id, fps=1. / timer.average_time
            )
        else:
            timer.toc()
            online_im = img_info['raw_img']

        # result_image = predictor.visual(outputs[0], img_info, predictor.confthre)
        if args.save_result:
            timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
            save_folder = osp.join(vis_folder, timestamp)
            os.makedirs(save_folder, exist_ok=True)
            cv2.imwrite(osp.join(save_folder, osp.basename(img_path)), online_im)

        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))

        # ch = cv2.waitKey(0)
        # if ch == 27 or ch == ord("q") or ch == ord("Q"):
        #     break

    if args.save_result:
        res_file = osp.join(vis_folder, f"{timestamp}.txt")
        with open(res_file, 'w') as f:
            f.writelines(results)
        logger.info(f"save results to {res_file}")


#图片流
def imageflow_demo(predictor, vis_folder, current_time, args):
    cap = cv2.VideoCapture(args.path if args.demo == "video" else args.camid)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    save_folder = osp.join(vis_folder, timestamp)
    os.makedirs(save_folder, exist_ok=True)
    if args.demo == "video":
        save_path = osp.join(save_folder, args.path.split("/")[-1])
    else:
        save_path = osp.join(save_folder, "camera.mp4")
    logger.info(f"video save_path is {save_path}")
    vid_writer = cv2.VideoWriter(
        save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
    )

    #定义追踪器
    tracker = BYTETracker(args, frame_rate=30)
    timer = Timer()
    frame_id = 0
    results = []
    # #视频处理
    # Note = open(r"C:\Users\lenovo\Desktop\lenoutput.txt", mode='w')
    i=0
    #过道计数
    count_channel = 0
    count_channel_copy = 0

    # 滚动数组统计进出
    shopin = 0
    shopout = 0

    lastchannel = []
    lastshop = []

    curshop = []
    curchannel = []

    output_xy=[]
    while True:
        #循环处理每一帧
        if frame_id % 20 == 0:
            #输出处理日志
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))
        ret_val, frame = cap.read()
        if ret_val:
            outputs, img_info = predictor.inference(frame, timer)



            #outputs[0]为检测数据的tensor数组, outputs_list-->outputs[0]
            select_array = torch.zeros(outputs[0].shape[0], dtype=torch.bool)


            # outputs_list=[]
            for j in range(outputs[0].shape[0]):
                if(judge_area(outputs[0][j][0],outputs[0][j][1],outputs[0][j][2],outputs[0][j][3])):
                    select_array[j]=1

                    # append_list=[]
                    # for k in range(len(outputs[0][j])):
                    #     append_list.append(outputs[0][j][k])
                    # outputs_list.append(append_list)
            # , dtype = torch.float64
            # outputs_tensor=torch.tensor(outputs_list)
            # print(outputs[0])
            select_array_reshaped = select_array.reshape(select_array.shape[0], 1)
            outputs_tensor= torch.masked_select(outputs[0], select_array_reshaped).reshape(-1, 7)
            # print(outputs_tensor)
            # .reshape((count, 7))
            # outputs_tensor = torch.cat(outputs_list, dim=0)

            # 调试文件

            # Note = open(r"C:\Users\lenovo\Desktop\outputs_tensor1.txt", mode='a')
            # # Note = open(r"C:\Users\lenovo\Desktop\outputs_list1.txt", mode='a')
            # i = i + 1
            # Note.write(str(i) + "\n")
            # Note.write(str(outputs_tensor.size()) + "\n")
            # # Note.write(str(len(outputs)) + "\n")
            # Note.write(str(outputs_tensor) + "\n\n")
            # Note.close()

            # if outputs[0] is not None:
            if outputs_tensor is not None:
                #输出跟踪结果online_targets
                # online_targets = tracker.update(outputs[0], [img_info['height'], img_info['width']], exp.test_size)
                online_targets = tracker.update(outputs_tensor, [img_info['height'], img_info['width']], exp.test_size)

                online_tlwhs = []
                online_ids = []    #id列表
                online_scores = []
                #t每一个跟踪目标
                for t in online_targets:
                    tlwh = t.tlwh
                    tid = t.track_id
                    vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                    if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        online_scores.append(t.score)
                        results.append(
                            f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                        )
                timer.toc()

                # Note = open(r"C:\Users\lenovo\Desktop\id.txt", mode='a')
                # Note = open(r"C:\Users\lenovo\Desktop\outputs_list1.txt", mode='a')
                # i = i + 1
                # Note.write(str(i) + "\n")
                # Note.write(str(outputs_tensor.size()) + "\n")
                # Note.write(str(len(outputs)) + "\n")
                # Note.write(str(online_ids) + "\n\n")
                # Note.close()

                count_channel_copy=max(online_ids)
                # 5帧更新count_channel
                if (frame_id % 5 == 1):
                    if (count_channel < count_channel_copy):
                        count_channel = count_channel_copy

                #电子栏、过道等
                draw_area(frame)

                # online_tlwhs,online_ids所有跟踪结果,统计进出
                #转换坐标格式
                # online_tlwhs = []
                bbox_xyxy=[]

                for i, tlwh in enumerate(online_tlwhs):
                    x1, y1, w, h = tlwh
                    bbox_xyxy.append(tuple(map(int, (x1, y1, x1 + w, y1 + h))))
                    if (judge_area(x1, y1, x1 + w, y1 + h) == 11 or judge_area(x1, y1, x1 + w, y1 + h) == 1):
                        # 热力图中每一点
                        output_xy.append([int(x1+w/2), int(y1 + h)])

                #15帧统计一次
                if (frame_id % 15 == 1):
                    # 第二帧时,frameidx==1
                    if (frame_id == 1):
                        for idex, box in enumerate(bbox_xyxy):
                            if (judge_area(box[0], box[1], box[2], box[3]) == 22):
                                curchannel.append(online_ids[idex])
                            elif (judge_area(box[0], box[1], box[2], box[3]) == 11):
                                curshop.append(online_ids[idex])
                            else:
                                continue

                    # 第五帧时,frameidx==4
                    # elif(frame_idx==1):
                    else:
                        # 更新跟踪点
                        lastchannel = curchannel
                        lastshop = curshop

                        curchannel = []
                        curshop = []

                        for idex, box in enumerate(bbox_xyxy):
                            # 靠边界区域追踪
                            if (judge_area(box[0], box[1], box[2], box[3]) == 22):
                                curchannel.append(online_ids[idex])
                            elif (judge_area(box[0], box[1], box[2], box[3]) == 11):
                                curshop.append(online_ids[idex])
                            else:
                                continue

                #显示统计数据
                # 8帧检查
                if (frame_id % 15 == 1):
                    # 统计进
                    if (len(curshop) and len(lastchannel)):
                        for customer in curshop:
                            if (customer in lastchannel):
                                shopin += 1
                            else:
                                continue

                    # 统计出
                    if (len(curchannel) and len(lastshop)):
                        for customer in curchannel:
                            if (customer in lastshop):
                                shopout += 1
                            else:
                                continue


                num_in = f"In: {shopin}"
                num_out = f"Out: {shopout}"
                num_str2 = f"Channel: {count_channel}"
                cv2.putText(frame, num_in, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)
                cv2.putText(frame, num_out, (140, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)
                cv2.putText(frame, num_str2, (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)


                # 跟踪结果作图
                online_im = plot_tracking(
                    img_info['raw_img'], online_tlwhs, online_ids, frame_id=frame_id + 1, fps=1. / timer.average_time
                )
            else:
                timer.toc()
                online_im = img_info['raw_img']
            if args.save_result:
                vid_writer.write(online_im)
            # ch = cv2.waitKey(1)
            # if ch == 27 or ch == ord("q") or ch == ord("Q"):
            #     break
        else:
            break
        frame_id += 1
    # Note.close()

    #热力图转numpy
    output_xy=np.array(output_xy)
    np.save('output_xy.npy',output_xy)

    if args.save_result:
        res_file = osp.join(vis_folder, f"{timestamp}.txt")
        with open(res_file, 'w') as f:
            f.writelines(results)
        logger.info(f"save results to {res_file}")


def main(exp, args):
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    output_dir = osp.join(exp.output_dir, args.experiment_name)
    os.makedirs(output_dir, exist_ok=True)

    if args.save_result:
        vis_folder = osp.join(output_dir, "track_vis")
        os.makedirs(vis_folder, exist_ok=True)

    if args.trt:
        args.device = "gpu"
        # args.device = "cpu"
    # args.device = torch.device("cuda" if args.device == "gpu" else "cpu")
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    #定义模型
    model = exp.get_model().to(args.device)
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))
    model.eval()

    if not args.trt:
        if args.ckpt is None:
            ckpt_file = osp.join(output_dir, "best_ckpt.pth.tar")
        else:
            ckpt_file = args.ckpt
        logger.info("loading checkpoint")
        ckpt = torch.load(ckpt_file, map_location="cpu")
        # load the model state dict
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    #注释half
    # model = model.float()
    if args.fp16:
        # model = model.half()  # to FP16
        model = model.float()

    if args.trt:
        assert not args.fuse, "TensorRT model is not support model fusing!"
        trt_file = osp.join(output_dir, "model_trt.pth")
        assert osp.exists(
            trt_file
        ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
        logger.info("Using TensorRT to inference")
    else:
        trt_file = None
        decoder = None

    #定义检测器
    predictor = Predictor(model, exp, trt_file, decoder, args.device, args.fp16)
    current_time = time.localtime()
    if args.demo == "image":
        image_demo(predictor, vis_folder, current_time, args)
    elif args.demo == "video" or args.demo == "webcam":
        #predictor检测器
        imageflow_demo(predictor, vis_folder, current_time, args)


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)

    main(exp, args)


#运行命令
# python tools/demo_track.py video --fp16 --fuse --save_result