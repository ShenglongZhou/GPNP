% demon compressed sensing problems using image data
clc; clear all; close all; warning off
addpath(genpath(pwd));

test  = 2;

switch test
    case 1; Img0 = im2double(imread('zhou.png')); 
    case 2; load col.mat
            Img0 = im2double(col);
end  
nsam     = 40;
sigma    = 0.05;
s        = 1500;
d        = 1 + 2*(test==2);
[d1,d2]  = size(Img0(:,:,1));
xdd      = zeros(d1,d2,d);

for k   = 1 : d
    Img = Img0(:,:,k);
    [A,At,b,be,xe,Ae,out] = getrealdata(Img,nsam,sigma,0);
    [d1,d2]   = size(out.I);
    n         = d2^2; 
    m         = length(b);  
    t         = tic;
    out0      = GPNPcs(n,s,b,A,At); 
    x         = out0.x;
    time      = toc(t);
    xd        = reshape(x,[d1 d2]);  
    xd        = out.W(xd) +  out.Ibar;   
    xdd(:,:,k)= xd;
    snr       = norm(out.I-xd,'fro')^2;           
end

snr   = 10*log10(d1*d2*d./snr);
figure('Renderer', 'painters', 'Position', [900 600 450 230])
for i   = 0:1
   sub  = subplot(1,2,i+1); 
   if i == 0,  imagesc(Img0); else; imagesc(xdd); end
   colormap(gray);
   pos  = get(sub, 'Position'); 
   ax   = gca; axis(ax,'off');
   if i  == 0
       title('Original Image')
   else
       title(ax,['GPNP: PSNR = ',num2str(snr(i),'%2.2f')]); 
   end
   set(sub, 'Position',pos+[-0.05*(2-i),-0.05,0.12,0.05] )
   ax.XLabel.Visible = 'on';
end
