from PIL import Image , ImageDraw
from image_tuple import create

num_img = 10000
img_size = (25,25)


def main(x :str) -> None:

    ev = ""
    for i in range(num_img) :
        img = Image.new("L",img_size,"white")
        if i % 2 == 1 :
            img.save("DATA_0/empty"+ev+"/img"+str(i)+".png")
        else :
            draw = ImageDraw.Draw(img,"L")
            draw.rectangle(create(0,img_size[0]-1),0)
            img.save("DATA_0/notempty"+ev+"/img"+str(i)+".png")
    print(x)
    
if __name__ == "__main__" :
    main(f"we create {num_img} images with the following size {img_size}")