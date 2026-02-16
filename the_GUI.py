from tkinter import *
from tkinter import ttk
from numpy_ai import main
import matplotlib.pyplot as plt
from PIL import Image , ImageTk

output_list : list[list[int]] = [[0,0]]
output_exp : str = ""
# back end

def check_valid_input(value , to_type_func) :
    try :
        return to_type_func(value)
    except :
        return None

def clr() :
    global output_exp
    output_exp = ""
    text_in_lable.set("lr(t+1) = ")

def update_to_sett() :
    remove_lr() ; remove_resu()
    frm_sett.place(x=0,rely=0.05,relheight=0.3,relwidth=1)
def remove_sett() :
    frm_sett.place_forget()

def update_to_resu() :
    remove_lr() ; remove_sett()
    frm_resu.place(x=0 , rely=0.05 , relheight=0.95 , relwidth=1)
def remove_resu() :
    frm_resu.place_forget()

def update_to_lr() :
    remove_sett() , remove_resu()
    frm_lr_left.place( x=0 , rely=0.05 , relheight=0.95 , relwidth=0.3)
    frm_lr_right.place( relx=0.3 , rely=0.05 , relheight=0.95 , relwidth=0.7)
def remove_lr() :
    frm_lr_left.place_forget()
    frm_lr_right.place_forget()

def add_tex(text_exp : str , text_lab : str) :
    global output_exp
    output_exp += text_exp
    print(output_exp)
    text_in_lable.set(text_in_lable.get()+ " " + text_lab)

#def var_(num : int, var : str) -> None:
#    global output_list , btn_right , text_in_lable
#    if len(output_list[len(output_list)-1]) == 2:
#        output_list.append([num])
#        text_in_lable.set(text_in_lable.get()+ " " + var)
#        print(output_list)
#
#def operation_(num : int, opp : str) -> None:
#    global output_list , btn_right
#    if len(output_list[len(output_list)-1]) == 1:
#        output_list[len(output_list)-1].append(num)
#        text_in_lable.set(text_in_lable.get()[:11]+"( "+text_in_lable.get()[11:]+" ) " + opp)
def create_resu(plot) :
    plt.plot(plot["batch_list"],plot["loss_list"]) ; plt.plot(plot["batch_list"],plot["acc_list"])
    plt.plot(plot["batch_list"],plot["lr_list"])
    plt.savefig("plot.png")

    Widget_resu[6].configure(text=str(plot["cv"]))
    Widget_resu[7].configure(text=str(plot["batch_num"]))
    Widget_resu[8].configure(text=str(plot["batch_s"]))
    Widget_resu[9].configure(text=str(plot["l_acc"]))
    Widget_resu[10].configure(text=str(plot["b_rea"]))
    Widget_resu[11].configure(text=text_in_lable.get())
    img = ImageTk.PhotoImage(Image.open("plot.png"))
    Widget_resu[12].configure(image=img)
def start() :
    a = check_valid_input(Widget_sett[0].get(),float)
    b = check_valid_input(Widget_sett[1].get(),int)
    c = check_valid_input(Widget_sett[2].get(),int)
    if a is not None and b is not None and c is not None :
        plot = main(lr=a,
            size_batch=b,
            number_of_iterations=c,
            exp=output_exp, update_lr=True)
    elif a is not None and b is not None:
        plot = main(lr=a,
          size_batch=b,
        exp=output_exp, update_lr=True)
    elif a is not None and c is not None :
        plot = main(lr=a,
            number_of_iterations=c,
            exp=output_exp, update_lr=True)
    elif b is not None and c is not None :
        plot = main(size_batch=b,
            number_of_iterations=c,
            exp=output_exp, update_lr=True)
    elif a is not None:
        plot = main(lr=a,
            exp=output_exp, update_lr=True)
    elif b is not None:
        plot = main(size_batch=b,
            exp=output_exp, update_lr=True)
    elif c is not None :
        plot = main(number_of_iterations=c,
            exp=output_exp, update_lr=True)
    else :
        plot = main(exp=output_exp, update_lr=True)
    create_resu(plot)
    update_to_resu()
# front end
root = Tk()

text_in_lable = StringVar(value="lr(t+1) = ")
root.geometry('900x600')

frm_task_bar = ttk.Frame(root)

frm_lr_left = ttk.Frame(root, padding=2)
frm_lr_right = ttk.Frame(root, padding=2)

frm_sett = ttk.Frame(root)
frm_resu = ttk.Frame(root)

frm_task_bar.place( x=0 , rely=0 , relheight=0.05 , relwidth=1)

frm_task_bar.grid_rowconfigure(0,weight=1 ,uniform="grid")
frm_task_bar.grid_columnconfigure((0,1,2),weight=1 , uniform="grid")

frm_lr_left.grid_rowconfigure((0,1,2,3,4,5,6,7),weight=1 ,uniform="grid")
frm_lr_left.grid_columnconfigure((0,1,2),weight=1 , uniform="grid")
frm_lr_right.grid_rowconfigure((0,1,2,3,4,5),weight=1 ,uniform="grid")
frm_lr_right.grid_columnconfigure((0,1,2),weight=1 , uniform="grid")

frm_sett.grid_rowconfigure((0,1),weight=1 ,uniform="grid")
frm_sett.grid_columnconfigure((0,1,2),weight=1 , uniform="grid")

frm_resu.grid_rowconfigure((0,1,2,3,4,5,6,7,8),weight=1 ,uniform="grid")
frm_resu.grid_columnconfigure((0,1,2,3,4,5,6,7,8),weight=1 , uniform="grid")

btn_task = [
    ttk.Button(frm_task_bar , text="lr_func" , command=update_to_lr).grid(row=0 , column=0 , sticky="news"),
    ttk.Button(frm_task_bar , text="setting" , command=update_to_sett).grid(row=0 , column=1 , sticky="news"),
    ttk.Button(frm_task_bar , text="result" , command=update_to_resu).grid(row=0 , column=2 , sticky="news"),
]
#lr_func
btn_lr_func = [
    # lable
    ttk.Label(frm_lr_right, background="red" , textvariable=text_in_lable ,
            font=("Arial",20)).grid(row=0 , column=0 , sticky='nswe' ,rowspan=2,columnspan=3),
    # variable 
    ttk.Button(frm_lr_right, text="lr(0)"    , command=lambda:add_tex("a","lr(0)" )),
    ttk.Button(frm_lr_right, text="the batch number", command=lambda:add_tex("b","b_n") ),
    ttk.Button(frm_lr_right, text="the batch size"  , command=lambda:add_tex("c","b_z")),
    ttk.Button(frm_lr_right, text="lr(t)"    , command=lambda:add_tex("x","lr(t)" )),
    ttk.Button(frm_lr_right, text="loss(t)", command=lambda:add_tex("y","loss(t)") ),
    ttk.Button(frm_lr_right, text="loss(t+1)"  , command=lambda:add_tex("z","loss(t+1)")),
    # operations
    ttk.Button(frm_lr_right, text="+" , command=lambda:add_tex("+","+")),
    ttk.Button(frm_lr_right, text="-" , command=lambda:add_tex("-","-")),
    ttk.Button(frm_lr_right, text="*" , command=lambda:add_tex("*","*")),
    ttk.Button(frm_lr_right, text="/" , command=lambda:add_tex("/","/")),
    ttk.Button(frm_lr_right, text="train" , command=start),
    ttk.Button(frm_lr_right, text="Quit" , command=root.destroy),
    #left
    ttk.Button(frm_lr_left,text='Mod' , command=lambda:add_tex('%','Mod')),
    ttk.Button(frm_lr_left,text='Max' , command=lambda:add_tex('Max','Max')),
    ttk.Button(frm_lr_left,text='Min' , command=lambda:add_tex('Min','Min')),

    ttk.Button(frm_lr_left,text='floor' , command=lambda:add_tex('floor','floor')),
    ttk.Button(frm_lr_left,text='sign' , command=lambda:add_tex('sign','sign')),
    ttk.Button(frm_lr_left,text='abs' , command=lambda:add_tex('abs','abs')),

    ttk.Button(frm_lr_left,text='(' , command=lambda:add_tex('(','(')),
    ttk.Button(frm_lr_left,text='^' , command=lambda:add_tex('**','^')),
    ttk.Button(frm_lr_left,text=')' , command=lambda:add_tex(')',')')),

    ttk.Button(frm_lr_left,text='1' , command=lambda:add_tex('1','1')),
    ttk.Button(frm_lr_left,text='2' , command=lambda:add_tex('2','2')),
    ttk.Button(frm_lr_left,text='3' , command=lambda:add_tex('3','3')),
    ttk.Button(frm_lr_left,text='4' , command=lambda:add_tex('4','4')),
    ttk.Button(frm_lr_left,text='5' , command=lambda:add_tex('5','5')),
    ttk.Button(frm_lr_left,text='6' , command=lambda:add_tex('6','6')),
    ttk.Button(frm_lr_left,text='7' , command=lambda:add_tex('7','7')),
    ttk.Button(frm_lr_left,text='8' , command=lambda:add_tex('8','8')),
    ttk.Button(frm_lr_left,text='9' , command=lambda:add_tex('9','9')),
    ttk.Button(frm_lr_left,text='0' , command=lambda:add_tex('0','0')),
    ttk.Button(frm_lr_left,text='.' , command=lambda:add_tex('.','.')),
    ttk.Button(frm_lr_left,text='clr' , command=clr),
    ttk.Button(frm_lr_left,text=',' , command=lambda:add_tex(',',','))
]
grid_params_lr_func = [
    # right
    (2, 0, "nswe"), (2, 1, "nswe"), (2, 2, "nswe"),
    (3, 0, "nswe"), (3, 1, "nswe"), (3, 2, "nswe"),
    (4, 0, "nswe"), (4, 1, "nswe"), (4, 2, "nswe"),
    (5, 0, "nswe"), (5, 1, "nswe"), (5, 2, "nswe"),

    # left
    (0, 0, "nswe"), (0, 1, "nswe"), (0, 2, "nswe"),
    (1, 0, "nswe"), (1, 1, "nswe"), (1, 2, "nswe"),
    (2, 0, "nswe"), (2, 1, "nswe"), (2, 2, "nswe"),
    (3, 0, "nswe"), (3, 1, "nswe"), (3, 2, "nswe"),
    (4, 0, "nswe"), (4, 1, "nswe"), (4, 2, "nswe"),
    (5, 0, "nswe"), (5, 1, "nswe"), (5, 2, "nswe"),
    (6, 0, "nswe"), (6, 1, "nswe"), (6, 2, "nswe"),
    (7, 0, "nswe")
]
for i in range(len(grid_params_lr_func)) :
    btn = btn_lr_func[i+1]
    para = grid_params_lr_func[i]
    btn.grid(row=para[0],column=para[1],sticky=para[2])

Widget_sett = [
    ttk.Entry(frm_sett),
    ttk.Entry(frm_sett),
    ttk.Entry(frm_sett),

    ttk.Label(frm_sett,text="Lr(0)" , width=7).grid(row=0,column=0 , sticky="nsew"),
    ttk.Label(frm_sett,text="batch size" , width=3).grid(row=0,column=1 , sticky="nsew"),
    ttk.Label(frm_sett,text="number of itteration" , width=6).grid(row=0,column=2 , sticky="nsew"),
]
Widget_sett[0].grid(row=1,column=0,sticky="news")
Widget_sett[1].grid(row=1,column=1,sticky="news")
Widget_sett[2].grid(row=1,column=2,sticky="news")

Widget_resu = [
    ttk.Label(frm_resu , text="C_V").grid(row=0 , column=0 , rowspan=1 , columnspan=2 , sticky="nswe"),
    ttk.Label(frm_resu , text="batch number").grid(row=1 , column=0 , rowspan=1 , columnspan=2 , sticky="nswe"),
    ttk.Label(frm_resu , text="batch size").grid(row=2 , column=0 , rowspan=1 , columnspan=2 , sticky="nswe"),
    ttk.Label(frm_resu , text="last acc").grid(row=3 , column=0 , rowspan=1 , columnspan=2 , sticky="nswe"),
    ttk.Label(frm_resu , text="reach best at batch").grid(row=4 , column=0 , rowspan=1 , columnspan=2 , sticky="nswe"),
    ttk.Label(frm_resu , text="schedual Method").grid(row=5 , column=0 , rowspan=1 , columnspan=2 , sticky="nswe"),

    ttk.Label(frm_resu , text="" , background="#34c7e9"),
    ttk.Label(frm_resu , text="" , background="#34c7e9"),
    ttk.Label(frm_resu , text="" , background="#34c7e9"),
    ttk.Label(frm_resu , text="" , background="#34c7e9"),
    ttk.Label(frm_resu , text="" , background="#34c7e9"),
    ttk.Label(frm_resu , text="" , background="#34c7e9"),
    ttk.Label(frm_resu)
]
Widget_resu[6].grid(row=0 , column=2 , rowspan=1 , columnspan=7 , sticky="nswe")
Widget_resu[7].grid(row=1 , column=2 , rowspan=1 , columnspan=7 , sticky="nswe")
Widget_resu[8].grid(row=2 , column=2 , rowspan=1 , columnspan=7 , sticky="nswe")
Widget_resu[9].grid(row=3 , column=2 , rowspan=1 , columnspan=7 , sticky="nswe")
Widget_resu[10].grid(row=4 , column=2 , rowspan=1 , columnspan=7 , sticky="nswe")
Widget_resu[11].grid(row=5 , column=2 , rowspan=1 , columnspan=7 , sticky="nswe")
Widget_resu[12].grid(row=6 , column=2 , rowspan=3 , columnspan=7 , sticky="nswe")
update_to_lr()
root.mainloop()