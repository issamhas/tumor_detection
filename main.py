from PIL import Image
import os, sys
#constants
_imgPath = 'bc_photos/mdb'
#image = Image.open('bc_photos/mdb322.pgm')
#image.save('bc_photos/mdb322.jpg')
#image.delete('bc_photos/mdb010.jpg')

#os.remove('bc_photos/mdb005.jpg')

def convert_to_jpg(path):

    for i in range(321):
        if i < 9:
            img_path = _imgPath+'00'+str(i+1)
        elif i<99:
            img_path = _imgPath+'0'+str(i+1)
        else:
            img_path = _imgPath+str(i+1)

        img = Image.open(img_path+'.pgm')
        img.save(img_path+".jpg")
        #os.remove(img_path+'.pgm')
#only convert once.
#convert_to_jpg(_imgPath)

#get list of filenames for tensorflow input
def get_file_names(path):
    filenames = []
    for i in range(322):
        if i < 9:
            temp_path = path+'00'+str(i+1)
        elif i<99:
            temp_path = path+'0'+str(i+1)
        else:
            temp_path = path+str(i+1)

        filenames.append(temp_path+'.jpg')
    

    print(filenames)

    #return filenames
	



get_file_names(_imgPath)
