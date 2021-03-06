close all
tic
load('transmit')
% file_to_use='../../data/dataset-iteration-fiber/0-3ed/train_spikes.bin' 
% disp(file_to_use)

ff2OUT=0;
TEST=0;
flag_offset_eval_examples=0;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Change NUMEVAL_SAMPLES in the next line to NUMEVAL_SAMPLES=20000 for the
%larger dataset provided
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
NUMEVAL_SAMPLES=1000;

%% The remaining parameters 
NUMEVAL_SAMPLES2=NUMEVAL_SAMPLES;
NUM=NUMEVAL_SAMPLES2;
HEIGHT=51;
WIDTH=51;
dim=100;200
HEIGHT2=dim;
WIDTH2=dim;
len2=dim-1;
intensity=0;
%% hooks
COR_avg=[];
PSNR_avg=[];
MSE_avg=[];
COR_avg_FOV=[];
COR_avg_intensity=[];
COR_avg_intensity_FOV=[];
COR_avg_G=[];
COR_avg_intensity_G=[];

%% body of the code starts here

n1=floor(NUMEVAL_SAMPLES/NUMEVAL_SAMPLES2);
res=NUMEVAL_SAMPLES-n1*NUMEVAL_SAMPLES2;
B=zeros(HEIGHT,WIDTH,NUMEVAL_SAMPLES2);
D=zeros(HEIGHT2,WIDTH2,NUMEVAL_SAMPLES2);

for i=1:n1
    str=['stimuli.bin'];
    str3=['targets.bin'];

    V=(1:NUMEVAL_SAMPLES2);
    V2=(1:NUMEVAL_SAMPLES2);
    

    
    f=fopen(str,'r'); 
    A=fread(f,HEIGHT*WIDTH*NUMEVAL_SAMPLES2,'uint8');
    temp=reshape(A,[NUMEVAL_SAMPLES2,HEIGHT,WIDTH]);
    B(:,:,V)=permute(temp,[2,3,1]);

    fclose(f);

    gg=fopen(str3,'r'); 
    A2=fread(gg,HEIGHT2*WIDTH2*NUMEVAL_SAMPLES2,'uint8');
    temp=reshape(A2,[NUMEVAL_SAMPLES2,HEIGHT2,WIDTH2]);
    D(:,:,V2)=permute(temp,[2,3,1]);

    fclose(gg);


B=B/255;
% B=2*B-1;
B=double(B);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[ff1,~]=clip_mask(holo_params.freq.maskc);
[ff2,~]=clip_mask(slm_params.freq.maskc);

if ff2OUT==1
    output=zeros(size(ff1,1),size(ff1,2),NUMEVAL_SAMPLES2);
    len1=size(ff1,1);
    v=((floor((len1-len2)/2)):(floor((len1+len2)/2)))+1;
else
    output=zeros(size(holo_params.freq.maskc,1),size(holo_params.freq.maskc,2),NUMEVAL_SAMPLES2);
    len1=size(holo_params.freq.maskc,1);
    v=(floor((len1-len2)/2)):(floor((len1+len2)/2));
end



for o=1:NUMEVAL_SAMPLES2
    o;

 X2_exp=B(:,:,o);
 X4=fftshift2(fft2(X2_exp));
 X5=mask(X4,ff2);
 X5_1=T*X5;
 if ff2OUT==1
   X5_2=unmask(X5_1,ff1);
 else  
   X5_2=unmask(X5_1,holo_params.freq.maskc); 
 end
 
  X5_3=ifft2(ifftshift2(X5_2));
  output(:,:,o)=X5_3;
end


if intensity==1
     OUT2=((abs(output(v,v,:))).^2)*255./max(max(((abs(output(v,v,:))).^2),[],1),[],2);
else
     OUT2=((abs(output(v,v,:))).^1)*255./max(max(((abs(output(v,v,:))).^1),[],1),[],2);
end

OUT3=zeros(size(OUT2,1),size(OUT2,2),NUM);

for ss=1:NUM  
    OUT3(:,:,ss)=(imadjust(OUT2(:,:,ss)/255,[100/255,255/255]))*255;  
end
 
    cor2=squeeze(my_corr(D(:,:,1:NUMEVAL_SAMPLES2),OUT2(:,:,1:NUMEVAL_SAMPLES2)));
    cor2_intensity=squeeze(my_corr(D(:,:,1:NUMEVAL_SAMPLES2),OUT2(:,:,1:NUMEVAL_SAMPLES2).^2));

    
% 
% %  
% if i<3
% cc=clock;
% eee=double(OUT2(:,:,2).^2);
% eee=uint8(255*eee/max(eee(:)));
% imwrite(eee,['Output_' num2str(cc(4:5)) '.png']);
% pause(1)
% 
% 
% cc=clock;
% eee=double(D(:,:,2));
% eee=uint8(255*eee/max(eee(:)));
% imwrite(eee,['Target_' num2str(cc(4:5)) '.png']);
% pause(1)
% 
% cc=clock;
% eee=B(:,:,2);
% eee=uint8(255*eee/max(eee(:)));
% imwrite(eee,['SLM_2D_input' num2str(cc(4:5)) '.png']);
% pause(1)

end

    COR_avg=[COR_avg mean(cor2(:))];

    COR_avg_intensity=[COR_avg_intensity mean(cor2_intensity(:))];
    MSE_avg=[MSE_avg mean(mean(mean((D-OUT2).^2)))/(255^2)];
    PSNR_avg=[PSNR_avg 10*log10(1/MSE_avg(end))];

COR_out=0;
MSE_out=0;
PSNR_out=0;

f7=fopen(file_to_use,'a+');
fwrite(f7,OUT2/255,'float32');
fclose(f7);



fid = fopen('AVG_corr_intensity.txt', 'a+');
fprintf(fid, '%d \n', mean(COR_avg_intensity));
fclose(fid);

corr_out = mean(COR_avg_intensity);
mse_out = mean(MSE_avg);
toc
