from PIL import Image
import os
 
def convert(dir,width):
    file_list = os.listdir(dir)
    #print(file_list)
    for filename in file_list:
        path = ''
        path = dir+filename
        im = Image.open(path)
        if im.mode=="P" or im.mode == "RGBA":
            im = im.convert('RGB')
        out = im.resize((width,width),Image.ANTIALIAS)
        print ("%s has been resized!"%filename)
        tmp_dir = r'C:\Users\Mr_Yao\Desktop\photos'
        resize_img_path = os.path.join(tmp_dir,filename)
        out.save(resize_img_path)
     
if __name__ == '__main__':
    #dir = raw_input('D:\dataset\panda:')
    convert(r'C:\Users\Mr_Yao\Desktop\photos/',416)
