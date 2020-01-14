import time
import os
from fastai import *
from fastai.vision import *
import fastai
from pathlib import Path
import torch
from flask import Flask, render_template, request, jsonify
import base64


#实例化对象
app = Flask(__name__)
# 设置开启web服务后，如果更新html文件，可以使更新立即生效
app.jinja_env.auto_reload = True
app.config['TEMPLATES_AUTO_RELOAD'] =True

#由GPU切换为CPU
fastai.basics.defaults.device = torch.device('cpu') 

#加载模型learner
path = Path(r'D:\DataSet\art')
learn = load_learner(path)

# 定义函数classId_to_className，把种类索引转换为种类名称
def classId_to_className(classId):
    classes = ['fushihui', 'shuimohua', 'yinxianghua']
    className = classes[classId]
    return className

classes = ['fushihui', 'shuimohua', 'yinxianghua']


# 使用模型对指定图片文件路径完成图像分类，返回值为预测的种类名称
def predict_image(img_path):
    #确保新传的数据和训练数据有相同的格式，相同的变形，相同的尺寸，相同的正则化
    data2 = ImageDataBunch.single_from_classes(path, classes, ds_tfms=get_transforms(), size=224)
    data2.normalize(imagenet_stats)   #正则化
    learn = create_cnn(data2, models.resnet34)
    #加载之前训练保存的模型参数
    learn.load('stage-2') 
    #对单张图片预测
    img = open_image(img_path)
    pred_class, pred_idx, outputs = learn.predict(img)
    print('对此图片路径 %s 的预测结果为 %s' %(img_path, pred_class))
    return str(pred_class)


# 访问首页时的调用函数
@app.route('/')
def index_page():
    return render_template('web_page_final.html')

# @app.route('/')
# def index():
#     img_path = 'C:/Users/Designer Su/Desktop/fastai/1-10_ArtClassification/static/3.jpg'
#     figfile = io.BytesIO(open(img_path, 'rb').read())
#     img = base64.b64encode(figfile.getvalue()).decode('ascii')
#     return render_template('web_page_final.html', img=img)


# 使用predict_image这个API服务时的调用函数  
@app.route("/predict_image", methods=['POST'])
def anyname_you_like():
    startTime = time.time()
    # 解析接收到的图片
    received_file = request.files['input_image']
    imageFileName = received_file.filename
    if received_file:
        # 保存接收的图片到指定文件夹
        received_dirPath = r'D:\DataSet\art\received_images'
        if not os.path.isdir(received_dirPath):
            os.makedirs(received_dirPath)
        imageFilePath = os.path.join(received_dirPath, imageFileName)
        received_file.save(imageFilePath)
        print('接收图片文件保存到此路径：%s' % imageFilePath)
        usedTime = time.time() - startTime
        print('接收图片并保存，总共耗时%.2f秒' % usedTime)
        # 对指定图片路径的图片做分类预测，并打印耗时，返回预测种类名称
        startTime = time.time()
        predict_className = predict_image(imageFilePath)
        usedTime = time.time() - startTime
        print('完成对接收图片的分类预测，总共耗时%.2f秒\n' % usedTime)
        return jsonify(predict_className=predict_className)


# 主函数        
if __name__ == "__main__":
    print('在开启服务前，先测试predict_image函数')
    imageFilePath = r'D:\DataSet\art\pre1.jpg'
    predict_className = predict_image(imageFilePath)
    app.run("127.0.0.1", port=5000)