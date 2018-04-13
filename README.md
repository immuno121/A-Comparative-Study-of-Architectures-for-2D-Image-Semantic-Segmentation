# Image-segmentation
IFCN:
1. take pool 5 output
2. let o pr pool 5 output: o->10X10
3. conv2DTranspose(o):o->22X22
4. o2= pool 4 output
5. crop(o,o2)
both are 20X20
6. add(o,o2)

____________________

1. take var1,var2,var3.
var1,conv layerafter pool 4;
do Conv2D(1,1,,21) to var1,var2,var3
size(var1)=size(var2)=size(var3)=20X20

2. o=add(o,var1),o=add(o,var2),o=add(o,var3)
o->20X20
temp20=0

3. take o2=pool3 output
o2=f3
o2:size=40X40
o2=COnv2D(1,1,21)

o=ConvTranpose2D(o)
o:size=44X44
crop(o,o2)

4. o=add(o,o2)

5. take var after pool 3
-> conv2D(var,1,2,3)
-> size(var1,var2,var3)=40X40


o=Add(o,var1)
o=Add(o,var2)
o=Add(o,var3)
temp4=o




__________________________________________
COntext


1. take f5(ouput of pool5)
o=f5:size=10X10

1.conv2D(ks=(3X3),depth=512,padding=same)(o)
batchnorm
relu
#output=o,
temp1=o


#input:o
2.conv2D(ks=(3X3),depth=512,padding=same)(o)
batchnorm
relu
temp2=o


3.conv2D(ks=(3X3),depth=512,padding=same)(o)
batchnorm
relu
 temp3=o
 
 
add all:
temp=add(temp1,temp2)



o=add(temp,temp3)

o:size=10X10

Conv2D(o):1X1X21


Conv2Dtranspose(o)-->o:size=20X20


add(o,temp20)

Conv2Dtranspose(o)-->o:size=40X40

add(o,temp40)


convTranspose2D(o)-> o:size=320X320# already done in the code


