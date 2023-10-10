import dlib
from imutils import face_utils
import cv2
import numpy as np
from PIL import Image
import random 

# --------------------------------
# 1.関数定義
# --------------------------------

#笑顔かどうかを判定（笑顔じゃなければ返り値 >= 0 になる）
def is_smile(landmark) :
    mouth_line = landmark[55 - 1][0] - landmark[49 - 1][0]
    left_face_line = (landmark[8 - 1][0] - landmark[7 - 1][0]) / 2 + landmark[7 - 1][0]
    right_face_line = (landmark[11 - 1][0] - landmark[10 - 1][0]) / 2 + landmark[10 - 1][0]
    face_line = right_face_line - left_face_line
    smile_digit = mouth_line - face_line
    return smile_digit

#笑顔じゃない人に笑顔の画像を貼り付ける
def paste_img(img, landmark, face_list):
    #face_listからランダムに画像を選ぶ
    face_str = ''.join(random.sample(face_list, 1))
    face_list.remove(face_str)
    #face_imgの画像読み込み
    face_img = cv2.imread(face_str, cv2.IMREAD_UNCHANGED)
    #face_imgのサイズ変更
    face_width = int((landmark[17 - 1][0] - landmark[1 - 1][0]) * 2)
    face_height = int((landmark[9 - 1][1] - landmark[27 - 1][1]) * 2.5)
    face_img = cv2.resize(face_img, (face_width, face_height))
    #画像の幅、高さ
    height, width, ch = face_img.shape
    #画像の位置調整
    cut_width = width // 2
    cut_height = height // 2
    #画像を重ねる座標指定
    point = (landmark[29 -1][0] - cut_width, landmark[29 -1][1] - cut_height)

    image = CvOverlayImage.overlay(img, face_img, point)
    return image


    

#2枚の画像を重ね合わせる
class CvOverlayImage(object):
    """
    [summary]
      OpenCV形式の画像に指定画像を重ねる
    """

    def __init__(self):
        pass

    @classmethod
    def overlay(
            cls,
            cv_background_image,
            cv_overlay_image,
            point,
    ):
        """
        [summary]
          OpenCV形式の画像に指定画像を重ねる
        Parameters
        ----------
        cv_background_image : [OpenCV Image]
        cv_overlay_image : [OpenCV Image]
        point : [(x, y)]
        Returns : [OpenCV Image]
        """
        overlay_height, overlay_width = cv_overlay_image.shape[:2]

        # OpenCV形式の画像をPIL形式に変換(α値含む)
        # 背景画像
        cv_rgb_bg_image = cv2.cvtColor(cv_background_image, cv2.COLOR_BGR2RGB)
        pil_rgb_bg_image = Image.fromarray(cv_rgb_bg_image)
        pil_rgba_bg_image = pil_rgb_bg_image.convert('RGBA')
        # オーバーレイ画像
        cv_rgb_ol_image = cv2.cvtColor(cv_overlay_image, cv2.COLOR_BGRA2RGBA)
        pil_rgb_ol_image = Image.fromarray(cv_rgb_ol_image)
        pil_rgba_ol_image = pil_rgb_ol_image.convert('RGBA')

        # composite()は同サイズ画像同士が必須のため、合成用画像を用意
        pil_rgba_bg_temp = Image.new('RGBA', pil_rgba_bg_image.size,
                                     (255, 255, 255, 0))
        # 座標を指定し重ね合わせる
        pil_rgba_bg_temp.paste(pil_rgba_ol_image, point, pil_rgba_ol_image)
        result_image = \
            Image.alpha_composite(pil_rgba_bg_image, pil_rgba_bg_temp)

        # OpenCV形式画像へ変換
        cv_bgr_result_image = cv2.cvtColor(
            np.asarray(result_image), cv2.COLOR_RGBA2BGRA)

        return cv_bgr_result_image
 
   
    
# --------------------------------
# 2.前準備
# --------------------------------

# 顔検出ツールの呼び出し
face_detector = dlib.get_frontal_face_detector()
# 顔のランドマーク検出ツールの呼び出し
predictor_path = 'shape_predictor_68_face_landmarks.dat'
face_predictor = dlib.shape_predictor(predictor_path)

# 検出対象の画像の呼び込み
input_img = 'sample_img/magao_group.jpeg'
img = cv2.imread(input_img)
pre_img = img

# 処理高速化のためグレースケール化
img_gry = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#笑顔の切り抜き画像をリストに格納
face_list = ["face_img/gendo.png", "face_img/sinji.png", "face_img/historia.png", "face_img/kaguya.png", "face_img/miyuki.png", "face_img/otae.png", "face_img/sinpachi.png"]

# --------------------------------
# 3.メインコード
# --------------------------------

# 顔検出
faces = face_detector(img_gry, 1)

# 検出した全顔に対して処理
for face in faces:
    # 顔のランドマーク検出
    landmark = face_predictor(img_gry, face)
    # 処理高速化のためランドマーク群をNumPy配列に変換
    landmark = face_utils.shape_to_np(landmark)
    #笑顔かどうか判定
    smile_digit = is_smile(landmark)
    #笑顔じゃない人に笑顔の画像を貼り付ける
    if smile_digit <= 0:
        img = paste_img(img, landmark, face_list)
        

    '''
    #ランドマーク描画
    for (i, (x, y)) in enumerate(landmark):
        cv2.circle(img, (x, y), 2, (0, 255, 0), -1)
    '''
    

# --------------------------------
# 4.結果表示
# --------------------------------

cv2.imshow('input_img', pre_img)
cv2.imshow('output_img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
