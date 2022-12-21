function varargout = DIP_Project(varargin)
% DIP_PROJECT MATLAB code for DIP_Project.fig
%      DIP_PROJECT, by itself, creates a new DIP_PROJECT or raises the existing
%      singleton*.
%
%      H = DIP_PROJECT returns the handle to a new DIP_PROJECT or the handle to
%      the existing singleton*.
%
%      DIP_PROJECT('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in DIP_PROJECT.M with the given input arguments.
%
%      DIP_PROJECT('Property','Value',...) creates a new DIP_PROJECT or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before DIP_Project_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to DIP_Project_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help DIP_Project

% Last Modified by GUIDE v2.5 21-Dec-2022 15:09:26

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @DIP_Project_OpeningFcn, ...
                   'gui_OutputFcn',  @DIP_Project_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before DIP_Project is made visible.
function DIP_Project_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to DIP_Project (see VARARGIN)

% Choose default command line output for DIP_Project
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes DIP_Project wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = DIP_Project_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --------------------------------------------------------------------
function point_operation_Callback(hObject, eventdata, handles)
% hObject    handle to point_operation (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --------------------------------------------------------------------
function color_image_operation_Callback(hObject, eventdata, handles)
% hObject    handle to color_image_operation (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --------------------------------------------------------------------
function changing_the_image_lighting_color_Callback(hObject, eventdata, handles)
% hObject    handle to swap_g_to_b (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% --------------------------------------------------------------------
function swapping_image_channels_Callback(hObject, eventdata, handles)
% hObject    handle to swapping_image_channels (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --------------------------------------------------------------------
function eliminating_color_channels_Callback(hObject, eventdata, handles)
% hObject    handle to eliminating_color_channels (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --------------------------------------------------------------------
function addition_Callback(hObject, eventdata, handles)
x1 = imread('photo.jpg');
x2 = imread('photo2.jpg');

y1 = imresize(x1 , [400,400]);
y2 = imresize(x2 , [400,400]);
Result = imadd(y1,y2);

subplot(3,2,1),imshow(y1),title('Input1');
subplot(3,2,5),imshow(y2),title('Input2');
subplot(3,2,4),imshow(Result),title('Result');


% --------------------------------------------------------------------
function subtraction_Callback(hObject, eventdata, handles)
x1 = imread('photo.jpg');
x2 = imread('photo2.jpg');

y1 = imresize(x1 , [400,400]);
y2 = imresize(x2 , [400,400]);
Result = imsubtract(y1,y2);

subplot(3,2,1),imshow(y1),title('Input1');
subplot(3,2,5),imshow(y2),title('Input2');
subplot(3,2,4),imshow(Result),title('Result');


% --------------------------------------------------------------------
function division_Callback(hObject, eventdata, handles)
x1 = imread('photo.jpg');
x2 = imread('photo2.jpg');

y1 = imresize(x1 , [400,400]);
y2 = imresize(x2 , [400,400]);
Result = imdivide(y1,y2);

subplot(3,2,1),imshow(y1),title('Input1');
subplot(3,2,5),imshow(y2),title('Input2');
subplot(3,2,4),imshow(Result),title('Result');
%-------------------------------------------
function multiplication_Callback(hObject, eventdata, handles)
x1 = imread('photo.jpg');
x2 = imread('photo2.jpg');

y1 = imresize(x1 , [400,400]);
y2 = imresize(x2 , [400,400]);
Result = immultiply(y1,y2)

subplot(3,2,1),imshow(y1),title('Input1');
subplot(3,2,5),imshow(y2),title('Input2');
subplot(3,2,4),imshow(Result),title('Result');

% --------------------------------------------------------------------
function complement_Callback(hObject, eventdata, handles)
x1 = imread('photo.jpg');
Result = imcomplement(x1);

subplot(1,2,1),imshow(x1),title('Input');
subplot(1,2,2),imshow(Result),title('Result');

% --------------------------------------------------------------------
function average_Callback(hObject, eventdata, handles)
x1 = imread('photo.jpg');
x2 = imread('photo2.jpg');

y1 = imresize(x1 , [400,400]);
y2 = imresize(x2 , [400,400]);
Result = imadd(y1,y2);
avg = Result / 2;
subplot(3,2,1),imshow(y1),title('Input1');
subplot(3,2,5),imshow(y2),title('Input2');
subplot(3,2,4),imshow(avg),title('Result');

% --------------------------------------------------------------------
function Maximum1_Callback(hObject, eventdata, handles)
x1 = imread('photo.jpg');
x2 = imread('photo2.jpg');

y1 = imresize(x1 , [400,400]);
y2 = imresize(x2 , [400,400]);
Result = max(y1,y2);

subplot(3,2,1),imshow(y1),title('Input1');
subplot(3,2,5),imshow(y2),title('Input2');
subplot(3,2,4),imshow(Result),title('Result');

% --------------------------------------------------------------------
function eliminate_red_Callback(hObject, eventdata, handles)
x = imread('photo.jpg');
x2 = x;
x2(:,:,1) = 0;

subplot(2,2,3),imshow(x),title('Input');
subplot(2,2,4),imshow(x2),title('Result');


% --------------------------------------------------------------------
function eliminate_green_Callback(hObject, eventdata, handles)
x = imread('photo.jpg');
x2 = x;
x2(:,:,2) = 0;

subplot(2,2,3),imshow(x),title('Input');
subplot(2,2,4),imshow(x2),title('Result');


% --------------------------------------------------------------------
function eliminate_blue_Callback(hObject, eventdata, handles)
x = imread('photo.jpg');
x2 = x;
x2(:,:,3) = 0;

subplot(2,2,3),imshow(x),title('Input');
subplot(2,2,4),imshow(x2),title('Result');


% --------------------------------------------------------------------
function swap_r_to_g_Callback(hObject, eventdata, handles)
x = imread('photo.jpg');
R = x(:,:,1);
G = x(:,:,2);
B = x(:,:,3);

temp = R
R = G;
G = temp;

x2(:,:,1) = R;
x2(:,:,2) = G;
x2(:,:,3) = B;

subplot(2,2,3),imshow(x),title('Input');
subplot(2,2,4),imshow(x2),title('Result');
% --------------------------------------------------------------------
function swap_r_to_b_Callback(hObject, eventdata, handles)
x = imread('photo.jpg');
R = x(:,:,1);
G = x(:,:,2);
B = x(:,:,3);

temp = R
R = B;
B = temp;

x2(:,:,1) = R;
x2(:,:,2) = G;
x2(:,:,3) = B;

subplot(2,2,3),imshow(x),title('Input');
subplot(2,2,4),imshow(x2),title('Result');


% --------------------------------------------------------------------
function swap_g_to_b_Callback(hObject, eventdata, handles)
x = imread('photo.jpg');
R = x(:,:,1);
G = x(:,:,2);
B = x(:,:,3);

temp = G
G = B;
B = temp;

x2(:,:,1) = R;
x2(:,:,2) = G;
x2(:,:,3) = B;

subplot(2,2,3),imshow(x),title('Input');
subplot(2,2,4),imshow(x2),title('Result');


% --------------------------------------------------------------------
function change_red_Callback(hObject, eventdata, handles)
x = imread('photo.jpg');
ch1 = x(:,:,1);
ch2 = x(:,:,2);
ch3 = x(:,:,3);

ch1 = ch1 + 100;

Result = cat(3,ch1,ch2,ch3);

subplot(2,2,3),imshow(x),title('Input');
subplot(2,2,4),imshow(Result),title('Result');

% --------------------------------------------------------------------
function change_blue_Callback(hObject, eventdata, handles)
x = imread('photo.jpg');
ch1 = x(:,:,1);
ch2 = x(:,:,2);
ch3 = x(:,:,3);

ch3 = ch3 + 100;

Result = cat(3,ch1,ch2,ch3);

subplot(2,2,3),imshow(x),title('Input');
subplot(2,2,4),imshow(Result),title('Result');


% --------------------------------------------------------------------
function change_green_Callback(hObject, eventdata, handles)
x = imread('photo.jpg');
ch1 = x(:,:,1);
ch2 = x(:,:,2);
ch3 = x(:,:,3);

ch2 = ch2 + 100;

Result = cat(3,ch1,ch2,ch3);

subplot(2,2,3),imshow(x),title('Input');
subplot(2,2,4),imshow(Result),title('Result');


% --------------------------------------------------------------------
function Image_histogram_Callback(hObject, eventdata, handles)
% hObject    handle to Image_histogram (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --------------------------------------------------------------------
function histogram_stretching_Callback(hObject, eventdata, handles)
% hObject    handle to histogram_stretching (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --------------------------------------------------------------------
function histogram_equalization_Callback(hObject, eventdata, handles)
% hObject    handle to histogram_equalization (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --------------------------------------------------------------------
function s_grey_image_Callback(hObject, eventdata, handles)
x = imread('photo.jpg');
x2 = rgb2gray(x);

subplot(3,3,2);imshow(x),title('original RGB Image');
subplot(3,3,4);imshow(x2),title('Original Gray Image');
subplot(3,3,6);imhist(x2),xlabel('Gray Level'),ylabel('# of pixels'),title('Histogram');
subplot(3,3,7);stretch = imadjust(x2);imshow(stretch),title('Stretched Image');
subplot(3,3,9);imhist(stretch),xlabel('Gray Level'),ylabel('# of pixels'),title('Stretched Histogram');

% --------------------------------------------------------------------
function s_rgb_image_Callback(hObject, eventdata, handles)
x = imread('photo.jpg');
ch1 = x(:,:,1);
ch2 = x(:,:,2);
ch3 = x(:,:,3);
subplot(3,3,7);imshow(x);title('original');

subplot(3,3,1);r1 = imhist(ch1),bar(r1,'r');title('R_histo');
subplot(3,3,2);g1 = imhist(ch2),bar(g1,'g');title('G_histo');
subplot(3,3,3);b1 = imhist(ch3),bar(b1,'b');title('B_histo');

r2 = imadjust(ch1,[],[],1.7);
g2 = imadjust(ch2,[],[],1.7);
b2 = imadjust(ch3,[],[],1.7);

y = cat(3,r2,g2,b2);

subplot(3,3,4);R = imhist(r2),bar(R,'r');title('R_histo Gamma');
subplot(3,3,5);G = imhist(g2),bar(G,'g');title('G_histo Gamma');
subplot(3,3,6);B = imhist(b2),bar(B,'b');title('B_histo Gamma');

subplot(3,3,9);imshow(y);title('After Gamma Correction');
% --------------------------------------------------------------------
function e_grey_image_Callback(hObject, eventdata, handles)
x = imread('photo.jpg');
x2 = rgb2gray(x);
y = histeq(x2);
subplot(2,2,1);imshow(x2);
subplot(2,2,2);imhist(x2);
subplot(2,2,3);imshow(y);
subplot(2,2,4);imhist(y);
% --------------------------------------------------------------------
function e_rgb_image_Callback(hObject, eventdata, handles)
x = imread('photo.jpg');
ch1 = x(:,:,1);
ch2 = x(:,:,2);
ch3 = x(:,:,3);
subplot(3,3,7);imshow(x);title('original');

subplot(3,3,1);r1 = imhist(ch1),bar(r1,'r');title('R_histo');
subplot(3,3,2);g1 = imhist(ch2),bar(g1,'g');title('G_histo');
subplot(3,3,3);b1 = imhist(ch3),bar(b1,'b');title('B_histo');

r2 = histeq(ch1);
g2 = histeq(ch2);
b2 = histeq(ch3);

y = cat(3,r2,g2,b2);

subplot(3,3,4);R = imhist(r2),bar(R,'r');title('R_histo Gamma');
subplot(3,3,5);G = imhist(g2),bar(G,'g');title('G_histo Gamma');
subplot(3,3,6);B = imhist(b2),bar(B,'b');title('B_histo Gamma');

subplot(3,3,9);imshow(y);title('After Correction');


% --------------------------------------------------------------------
function neighborhood_processing_Callback(hObject, eventdata, handles)
% hObject    handle to neighborhood_processing (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --------------------------------------------------------------------
function image_restoration_Callback(hObject, eventdata, handles)
% hObject    handle to image_restoration (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --------------------------------------------------------------------
function Untitled_9_Callback(hObject, eventdata, handles)
% hObject    handle to Untitled_9 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --------------------------------------------------------------------
function edge_detection_Callback(hObject, eventdata, handles)
 image=imread('photo.jpg');
image=rgb2gray(image);
edge_p=edge(image,'prewitt');
edge_s=edge(image,'sobel');
edge_r=edge(image,'roberts');
subplot(2,2,1),imshow(image);title('original');
subplot(2,2,2),imshow(edge_p);title('prewitt operator')
subplot(2,2,3),imshow(edge_s);title('sobel operator')
subplot(2,2,4),imshow(edge_r);title('roberts operator')


% --------------------------------------------------------------------
function mathematical_morphology_Callback(hObject, eventdata, handles)
% hObject    handle to mathematical_morphology (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --------------------------------------------------------------------
function image_dilation_Callback(hObject, eventdata, handles)
im= imread('text.png');
               sq=ones(2,2);
               td=imdilate(im, sq);
              subplot(1,2,1),imshow(im)
              subplot(1,2,2),imshow(td)


% --------------------------------------------------------------------
function image_erosion_Callback(hObject, eventdata, handles)
im= imread('text.png');
           sq=ones(2,2);
           td=imerode(im, sq);
          subplot(1,2,1),imshow(im)
         subplot(1,2,2),imshow(td)


% --------------------------------------------------------------------
function image_opening_Callback(hObject, eventdata, handles)
im=imread('text.png');
im=double(im);
im=imnoise(im,'salt & pepper',.01);
sq=ones(2,2);
td=imopen(im, sq);
subplot(1,2,1),imshow(im)
subplot(1,2,2),imshow(td)


% --------------------------------------------------------------------
function boundary_extraction_Callback(hObject, eventdata, handles)
im=imread('photo.jpg');
im=double(im);
sq=ones(3,3);
td=imerode(im, sq);
in=abs(im-td);
td2=imdilate(im, sq);
ex=abs(td2-im);
mg=abs(td2-td);

subplot(2,2,1),imshow(im);title('binary image')
subplot(2,2,2),imshow(in);title('internal boundry')
subplot(2,2,3),imshow(ex);title('external boundry')
subplot(2,2,4),imshow(mg);title('morphological gradient')



% --------------------------------------------------------------------
function Thresholding_Callback(hObject, eventdata, handles)
% hObject    handle to Thresholding (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --------------------------------------------------------------------
function Salt_and_pepper_noise_Callback(hObject, eventdata, handles)
% hObject    handle to Salt_and_pepper_noise (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --------------------------------------------------------------------
function Gaussian_noise_Callback(hObject, eventdata, handles)
% hObject    handle to Gaussian_noise (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --------------------------------------------------------------------
function Linear_filter_Callback(hObject, ~, handles)
% hObject    handle to Linear_filter (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --------------------------------------------------------------------
function Non_linear_filters_Callback(hObject, eventdata, handles)
% hObject    handle to Non_linear_filters (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --------------------------------------------------------------------
function Maximum_Callback(hObject, eventdata, handles)
X=imread('photo.jpg');
x = rgb2gray(X);
y=colfilt(x,[3 3],'sliding' ,@max);
 subplot(2,1,1),imshow(x);title('Input');
 subplot(2,1,2), imshow(y);title('Result');


% --------------------------------------------------------------------
function Minimum_Callback(hObject, eventdata, handles)
X=imread('photo.jpg');
x = rgb2gray(X);
y=colfilt(x,[3 3],'sliding' ,@min);
 subplot(2,1,1),imshow(x);title('Input');
 subplot(2,1,2), imshow(y);title('Result');

% --------------------------------------------------------------------
function range_Callback(hObject, eventdata, handles)
X=imread('photo.jpg');
x = rgb2gray(X);
y=colfilt(x,[3 3],'sliding' ,@range);
 subplot(2,1,1),imshow(x);title('Input');
 subplot(2,1,2), imshow(y);title('Result');
 
% --------------------------------------------------------------------
function Most_frequent_values_Callback(hObject, eventdata, handles)
X=imread('photo.jpg');
x = rgb2gray(X);
y=colfilt(x,[3 3],'sliding' ,@mode);
 subplot(2,1,1),imshow(x);title('Input');
 subplot(2,1,2), imshow(y);title('Result');

% --------------------------------------------------------------------
function Rank_Order_Callback(hObject, eventdata, handles)
X=imread('photo.jpg');
x = rgb2gray(X);
y=ordfilt2(x,5,ones(3 ,3));
subplot(2,1,1),imshow(x),title('Input');
subplot(2,1,2), imshow(y),title('Result'); 


% --------------------------------------------------------------------
function Median_Callback(hObject, eventdata, handles)
X=imread('photo.jpg');
x = rgb2gray(X);
y=colfilt(x,[3 3],'sliding' ,@median);
 subplot(2,1,1),imshow(x);title('Input');
 subplot(2,1,2), imshow(y);title('Result');




% --------------------------------------------------------------------
function L_Average_filter_Callback(hObject, eventdata, handles)
rgb=imread('photo.jpg');subplot(1,2,1);imshow(rgb)
title('origonal image');
h=fspecial('average');
r=imfilter(rgb(:,:,1),h);
g=imfilter(rgb(:,:,2),h);
b=imfilter(rgb(:,:,3),h);
blur=cat(3,r,g,b);
subplot(1,2,2);
imshow(blur)
title('after filter');


% --------------------------------------------------------------------
function Laplacian_filter_Callback(hObject, eventdata, handles)
img=imread('photo.jpg');
c=rgb2gray(img);
%create  filter
f1=fspecial('laplacian');     
cf1=filter2(f1,c); subplot(2,1,1);imshow(c);title('Input');
subplot(2,1,2);imshow(cf1/255);title('Result');


% --------------------------------------------------------------------
function Image_averaging_Callback(hObject, eventdata, handles)
img=imread('photo.jpg');
x= rgb2gray(img);
[r c h]=size(x);
g=zeros(r ,c,10);
for i=1:10
    g(:,:,i)=imnoise(x,'gaussian');
 subplot(2,5,i),imshow(mat2gray(g(:,:,i)));
end
res=mat2gray(mean(g,3));
figure,imshow(res);title('Result');


% --------------------------------------------------------------------
function G_Average_filter_Callback(hObject, eventdata, handles)
img=imread('photo.jpg');
x= rgb2gray(img);
subplot(3,1,1)
imshow(x),title('Input'),y= imnoise(x,'gaussian');
subplot(3,1,2),imshow(y);title('Noise');
h=fspecial('average');
B2=imfilter(y,h);subplot(3,1,3);imshow(B2);title('Result');


% --------------------------------------------------------------------
function S_Average_filter_Callback(hObject, eventdata, handles)
img= imread('photo.jpg');
x= rgb2gray(img);
subplot(3,1,1)
imshow(x);title('Input');
y= imnoise(x,'salt & pepper',.05);
subplot(3,1,2);imshow(y);title('Noise Image');
h=fspecial('average',[7 7]);
B2=imfilter(y,h);
subplot(3,1,3);title('Average Filter');
imshow(B2);


% --------------------------------------------------------------------
function Median_filter_Callback(hObject, eventdata, handles)
img= imread('photo.jpg');
x= rgb2gray(img);
subplot(3,1,1)
imshow(x);title('Input');
y= imnoise(x,'salt & pepper',.05);
subplot(3,1,2);imshow(y);title('Noise Image');
B2=medfilt2(y,[3 3]);
subplot(3,1,3);title('Median Filter');
imshow(B2);


% --------------------------------------------------------------------
function Rank_order_filter_Callback(hObject, eventdata, handles)
img= imread('photo.jpg');
x= rgb2gray(img);
subplot(3,1,1)
imshow(x);title('Input');
y= imnoise(x,'salt & pepper',.05);
subplot(3,1,2);imshow(y);title('Noise Image');
B2=ordfilt2(y,5,ones(3,3));
subplot(3,1,3);imshow(B2);title('Rank order Filter');


% --------------------------------------------------------------------
function An_outlier_method_Callback(hObject, eventdata, handles)
img= imread('photo.jpg');
x= rgb2gray(img);
subplot(3,1,1)
imshow(x);title('Input');
y= imnoise(x,'salt & pepper',.05);
subplot(3,1,2);imshow(y);title('Noise Image');
f = [1/8 1/8 1/8; 1/8 0 1/8; 1/8 1/8 1/8];
im = im2double(y); imf=filter2(f,im);
[r,c]=size(x);
diff = abs(im-imf);
clean=zeros(r,c);
for i=1:r
    for j=1:c
        if diff(i,j)>.4
            clean(i,j)=imf(i,j);
        else
            clean(i,j)=im(i,j);
        end
    end
end
subplot(3,1,3),imshow(clean);title('clean');


% --------------------------------------------------------------------
function Basic_Global_Thresholding_Callback(hObject, eventdata, handles)
image1=imread('rice.png');
subplot(3,1,1);imhist(image1);
subplot(3,1,2);imshow(image1)
subplot(3,1,3);imshow(image1>130)



% --------------------------------------------------------------------
function Automatic_Thresholding_Callback(hObject, eventdata, handles)
f=imread('rice.png');
Theta = mean2(f); 
 done = 0;
 while ~done 
g = (f >=Theta);
m1=mean(f(g));
m2=mean(f(~g));
Th_next = 0.5*( m1+ m2);   
done = abs(Theta - Th_next) < 0.5;   
Theta = Th_next;   
 end 
 x=im2bw(f,Theta/255);
  figure, subplot(2,1,1), imshow(f)
 subplot(2,1,2),imshow(x)


% --------------------------------------------------------------------
function Adaptive_Thresholding_Callback(hObject, eventdata, handles)
c=imread('photo.jpg');
c=rgb2gray(c);
p1=c(:,1:64);
p2=c(:,65:128);
p3=c(:,129:192);
p4=c(:,193:256);
t1=im2bw(p1,graythresh(p1));
t2=im2bw(p2,graythresh(p2));
t3=im2bw(p3,graythresh(p3));
t4=im2bw(p4,graythresh(p4));
x=[t1 t2 t3 t4];
subplot(2,1,1),imshow(c);title('Input');
 subplot(2,1,2),imshow(x);title('Adaptive Thresholding');



% --------------------------------------------------------------------
function internal_boundary_Callback(hObject, eventdata, handles)
% hObject    handle to internal_boundary (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --------------------------------------------------------------------
function external_boundary_Callback(hObject, eventdata, handles)
% hObject    handle to external_boundary (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --------------------------------------------------------------------
function morphological_gradient_Callback(hObject, eventdata, handles)
% hObject    handle to morphological_gradient (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on selection change in popupmenu2.
function popupmenu2_Callback(hObject, eventdata, handles)
% hObject    handle to popupmenu2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns popupmenu2 contents as cell array
%        contents{get(hObject,'Value')} returns selected item from popupmenu2


% --- Executes during object creation, after setting all properties.
function popupmenu2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to popupmenu2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
