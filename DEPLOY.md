# install
  mkvirtualenv paddle
  python3 -m pip install paddlepaddle -i https://mirror.baidu.com/pypi/simple
  python3 -m pip install paddlepaddle-gpu -i https://mirror.baidu.com/pypi/simple
  pip3 install "paddleocr>=2.0.1"
  pip3 install -r requirements.txt

  pip3 install tensorflow==1.14.0

# run detector
## image
python3 tools/infer/predict_image.py --image_dir="/home/zhuhao/tmp/images/4_004.png" --det_model_dir="/home/zhuhao/myModel/paddle/ch_ppocr_server_v2.0_det_infer/" --draw_img_save_dir="/home/zhuhao/tmp/result" --det_db_box_thresh=0.3 --det_db_score_mode='slow' --use_dilation=True --det_limit_side_len=2400

## video
### server
python3 tools/infer/predict_video.py --det_model_dir="/home/zhuhao/myModel/paddle/ch_ppocr_server_v2.0_det_infer/" --video_file="/home/zhuhao/video/car/video_001.mp4"  --video_save_file="/home/zhuhao/video/car/result/video_001.mp4" --det_db_score_mode='slow' --use_dilation=True --det_limit_side_len=2400 --det_db_box_thresh=0.3

python3 tools/infer/predict_video.py --det_model_dir="/home/zhuhao/myModel/paddle/ch_ppocr_server_v2.0_det_infer/" --video_file="/home/zhuhao/video/car/video_002.mp4"  --video_save_file="/home/zhuhao/video/car/result/video_002.mp4" --det_db_score_mode='slow' --use_dilation=True --det_limit_side_len=2400 --det_db_box_thresh=0.3

python3 tools/infer/predict_video.py --det_model_dir="/home/zhuhao/myModel/paddle/ch_ppocr_server_v2.0_det_infer/" --video_file="/home/zhuhao/video/car/video_003.mp4"  --video_save_file="/home/zhuhao/video/car/result/video_003.mp4" --det_db_score_mode='slow' --use_dilation=True --det_limit_side_len=2400 --det_db_box_thresh=0.3

python3 tools/infer/predict_video.py --det_model_dir="/home/zhuhao/myModel/paddle/ch_ppocr_server_v2.0_det_infer/" --video_file="/home/zhuhao/video/car/video_004.mp4"  --video_save_file="/home/zhuhao/video/car/result/video_004.mp4" --det_db_score_mode='slow' --use_dilation=True --det_limit_side_len=2400 --det_db_box_thresh=0.3

python3 tools/infer/predict_video.py --det_model_dir="/home/zhuhao/myModel/paddle/ch_ppocr_server_v2.0_det_infer/" --video_file="/home/zhuhao/video/car/video_005.mp4"  --video_save_file="/home/zhuhao/video/car/result/video_005.mp4" --det_db_score_mode='slow' --use_dilation=True --det_limit_side_len=2400 --det_db_box_thresh=0.3

python3 tools/infer/predict_video.py --det_model_dir="/home/zhuhao/myModel/paddle/ch_ppocr_server_v2.0_det_infer/" --video_file="/home/zhuhao/video/car/video_006.mp4"  --video_save_file="/home/zhuhao/video/car/result/video_006.mp4" --det_db_score_mode='slow' --use_dilation=True --det_limit_side_len=2400 --det_db_box_thresh=0.3

python3 tools/infer/predict_video.py --det_model_dir="/home/zhuhao/myModel/paddle/ch_ppocr_server_v2.0_det_infer/" --video_file="/home/zhuhao/video/car/video_007.mp4"  --video_save_file="/home/zhuhao/video/car/result/video_007.mp4" --det_db_score_mode='slow' --use_dilation=True --det_limit_side_len=2400 --det_db_box_thresh=0.3

#### ch_ppocr_mobile_v2.0_det_prune_infer
python3 tools/infer/predict_video.py --det_model_dir="/home/zhuhao/myModel/paddle/ch_ppocr_mobile_v2.0_det_prune_infer/" --video_file="/home/zhuhao/video/car/SecurityCameraCatchesParkingLotIncident.mp4"  --video_save_file="/home/zhuhao/video/car/result/SecurityCameraCatchesParkingLotIncident.mp4" --det_db_score_mode='slow' --use_dilation=True --det_limit_side_len=2400

python3 tools/infer/predict_video.py --det_model_dir="/home/zhuhao/myModel/paddle/ch_ppocr_mobile_v2.0_det_prune_infer/" --video_file="/home/zhuhao/video/car/ClearPixCameraGroceryParkingLot.mp4"  --video_save_file="/home/zhuhao/video/car/result/ClearPixCameraGroceryParkingLot.mp4" --det_db_score_mode='slow' --use_dilation=True --det_limit_side_len=2400 --det_db_box_thresh=0.3

python3 tools/infer/predict_video.py --det_model_dir="/home/zhuhao/myModel/paddle/ch_ppocr_mobile_v2.0_det_prune_infer/" --video_file="/home/zhuhao/video/car/IonAlarmCCTVHDParkingLotCamera.mp4"  --video_save_file="/home/zhuhao/video/car/result/IonAlarmCCTVHDParkingLotCamera.mp4" --det_db_score_mode='slow' --use_dilation=True --det_limit_side_len=2400


===========================================
python3 tools/infer/predict_det.py --image_dir="/home/zhuhao/dataset/car_with_license_plate/images" --det_model_dir="/home/zhuhao/myModel/paddle/ch_ppocr_mobile_v2.0_det_prune_infer/" --draw_img_save_dir="/home/zhuhao/dataset/results/plate/ch_ppocr_mobile_v2.0_det_prune_infer"

python3 tools/infer/predict_det.py --image_dir="/home/zhuhao/dataset/car_with_license_plate/images" --det_model_dir="/home/zhuhao/myModel/paddle/ch_PP-OCRv2_det_slim_quant_infer/" --draw_img_save_dir="/home/zhuhao/dataset/results/plate/ch_PP-OCRv2_det_slim_quant_infer"

python3 tools/infer/predict_det.py --image_dir="/home/zhuhao/dataset/car_with_license_plate/images" --det_model_dir="/home/zhuhao/myModel/paddle/ch_PP-OCRv2_det_infer/" --draw_img_save_dir="/home/zhuhao/dataset/results/plate/ch_PP-OCRv2_det_infer"

python3 tools/infer/predict_det.py --image_dir="/home/zhuhao/dataset/car_with_license_plate/images" --det_model_dir="/home/zhuhao/myModel/paddle/ch_ppocr_mobile_v2.0_det_infer/" --draw_img_save_dir="/home/zhuhao/dataset/results/plate/ch_ppocr_mobile_v2.0_det_infer"

python3 tools/infer/predict_det.py --image_dir="/home/zhuhao/dataset/car_with_license_plate/images" --det_model_dir="/home/zhuhao/myModel/paddle/ch_ppocr_server_v2.0_det_infer/" --draw_img_save_dir="/home/zhuhao/dataset/results/plate/ch_ppocr_server_v2.0_det_infer"


=======================
python3 tools/infer/predict_det.py --image_dir="/home/zhuhao/dataset/tmp/images/car_001.jpg" --det_model_dir="/home/zhuhao/myModel/paddle/ch_PP-OCRv2_det_infer/"

python3 tools/infer/predict_det.py --image_dir="/home/zhuhao/dataset/tmp/images/car_001.jpg" --det_model_dir="/home/zhuhao/myModel/paddle/ch_ppocr_mobile_v2.0_det_infer/"

python3 tools/infer/predict_det.py --image_dir="/home/zhuhao/dataset/tmp/images/car_001.jpg" --det_model_dir="/home/zhuhao/myModel/paddle/ch_ppocr_mobile_v2.0_det_prune_infer/"

python3 tools/infer/predict_det.py --det_model_dir="/home/zhuhao/myModel/paddle/ch_PP-OCRv2_det_slim_quant_infer/" --image_dir="/home/zhuhao/dataset/tmp/images/car_001.jpg"

python3 tools/infer/predict_det.py --det_model_dir="/home/zhuhao/myModel/paddle/ch_ppocr_server_v2.0_det_infer/" --image_dir="/home/zhuhao/dataset/tmp/images/car_001.jpg"

