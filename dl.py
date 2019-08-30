import caffe
import cv2
import sys

def deploy(img_path):

    caffe.set_mode_gpu()
    MODEL_JOB_DIR = '/dli/data/digits/20190830-133052-cf23' 
    ARCHITECTURE = MODEL_JOB_DIR + '/' + 'deploy.prototxt'
    WEIGHTS = MODEL_JOB_DIR + '/' + 'snapshot_iter_270.caffemodel'
    
    net = caffe.Classifier(ARCHITECTURE,WEIGHTS ,
                           channel_swap=(2,1,0),
                           raw_scale=255,
                           image_dims=(256, 256))
                       

    input_image= caffe.io.load_image(img_path)
    input_image = cv2.resize(input_image, (256,256))
    mean_image = caffe.io.load_image('/dli/data/digits/20190830-132817-ddee/mean.jpg')
    input_image = input_image-mean_image

    
    prediction = net.predict([input_image])

    if prediction.argmax()==0:
        return "whale"
    else:
        return "not whale"

    
##Ignore this part    
if __name__ == '__main__':
    print(deploy(sys.argv[1]))
