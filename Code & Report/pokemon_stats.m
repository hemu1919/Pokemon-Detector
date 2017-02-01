function [ID, CP, HP, stardust, level, cir_center] = pokemon_stats (img, model)
% Please DO NOT change the interface
% INPUT: image; model(a struct that contains your classification model, detector, template, etc.)
% OUTPUT: ID(pokemon id, 1-201); level(the position(x,y) of the white dot in the semi circle); cir_center(the position(x,y) of the center of the semi circle)


% Replace these with your code
ID = getId(img,model.IdInfo);
CP=getCP(img,model.CpInfo,model.DigitInfo);
HP = getHP(img,model.HpInfo,model.DigitInfo);
stardust = getSD(img,model.SdInfo,model.DigitInfo);
cir_center=getCenter(img,model.CenterInfo);
level = getLevel(img,10,model.LevelInfo);
end

function cir_center = getCenter(img,model)
    tmp1=model.temp1_0;
    tmp2=model.temp1_3;
    [col,row,~]=size(img);
    cx=0.5*row;
    tmp1=imresize(tmp1,round([col*0.1577 row*0.2789]));
    tmp2=imresize(tmp2,round([col*0.0602 row*0.1175]));
    corr1=normxcorr2(rgb2gray(tmp1),rgb2gray(img));
    corr2=normxcorr2(rgb2gray(tmp2),rgb2gray(img));
    [ypeak1,~] = find(corr1==max(corr1(:)));
    [ypeak2,~] = find(corr2==max(corr2(:)));
    yoffSet1=ypeak1-size(tmp1,1);
    yoffSet2=ypeak2-size(tmp2,1);
    cy2=yoffSet2+0.0353*size(tmp2,1);
    cy1=yoffSet1+0.0648*size(tmp1,1);
    if(cy1 > cy2 && cy1 < 0.45*col)
        cy=cy1;
    elseif(cy2 > cy1 && cy2 < 0.45*col)
        cy=cy2;
    else
        cy=cy1;
    end
    cir_center=[cx cy];
end

function level = getLevel(img,th,model)
    step=1;
    y1=0;y2=0;
    tmp=model.temp1_0;
    gc=[];gr=[];
    while(step<=th)
        img_filt=imgaussfilt3(img,step);
        row=size(img_filt,1);
        if(size(size(img_filt),2)==2)
            [c, r]=imfindcircles(img_filt,round(0.0093*size(img_filt,2)),'ObjectPolarity','dark','Sensitivity',0.9659,'EdgeThreshold',0.1);
        else
            [c, r]=imfindcircles(img_filt,round(0.0093*size(img_filt,2)),'ObjectPolarity','bright','Sensitivity',0.9659,'EdgeThreshold',0.1);
        end
        if(step==1)
            tmp=imresize(tmp,round([size(img_filt,1)*0.1577 size(img_filt,2)*0.2789]));
            if(size(size(img_filt),2)==2)
                corr=normxcorr2(rgb2gray(tmp),img_filt);
            else
                corr=normxcorr2(rgb2gray(tmp),rgb2gray(img_filt));
            end
            [ypeak,~] = find(corr==max(corr(:)));
            y1=min((ypeak - 0.7880*size(tmp,1))+1,0.10*size(img_filt,2));
            y2=round(size(img_filt,1)/2)-0.1427*size(img_filt,1);
        end
        set=find(c(1:end,2) > y1);
        c=c(set,:);
        set=find(c(1:end,2) < y2);
        c=c(set,:);
        r=r(set);
        counts_ref=model.counts_temp;
        counts=zeros(size(c,1),1);
        for j = 1 : size(c,1)
            if(size(size(img_filt),2)==2)
                img1=imcrop(img_filt,[c(j,1)-r(j) c(j,2)-r(j) 2*r(j) 2*r(j)]);
                i_th=64;
            else
                i_th=240;
                img1=imcrop(rgb2gray(img_filt),[c(j,1)-r(j) c(j,2)-r(j) 2*r(j) 2*r(j)]);
            end
            [counts_temp,~]=imhist(img1);
            counts_temp=counts_temp/sum(counts_temp);
            [~,count]=max(counts_temp);
            if(count<i_th || img1(round(size(img1,1)/2),round(size(img1,2)/2)) < i_th)
                continue;
            end
            counts(j)=similarity(counts_temp',counts_ref');
        end
        [count,ind]=max(counts);
        if(size(c,1) ~= 0)
            gc(end+1,:)=[c(ind,:) count];
            gr(end+1)=r(ind);
        end
        step=step+1;
    end
    svm=model.svm;
    [~,ind]=max(gc(:,3));
    counts_temp=imhist(imcrop(rgb2gray(img_filt),[gc(ind,1)-gr(ind) gc(ind,2)-gr(ind) 2*gr(ind) 2*gr(ind)]));
    while(~predict(svm,counts_temp') && sum(gc(ind,3))~=0)
        gc(ind,3)=0;
        [~,ind]=max(gc(:,3));
        counts_temp=imhist(imcrop(rgb2gray(img_filt),[gc(ind,1)-gr(ind) gc(ind,2)-gr(ind) 2*gr(ind) 2*gr(ind)]));
    end
    level=gc(ind,1:2);
end

function id = getId(img,model)
    tmp=model.temp1_0;
    tmp=imresize(tmp,round([size(img,1)*0.1577 size(img,2)*0.2789]));
    corr=normxcorr2(rgb2gray(tmp),rgb2gray(img));
    [ypeak,~] = find(corr==max(corr(:)));
    row=size(img,2);
    x1=round(row*0.2);
    x2=round(row*0.8);
    y1=round((ypeak - size(tmp,1))-(0.2605*size(img,1)));
    y2=round(0.45*size(img,1));
    img1=imcrop(img,[x1 y1 x2-x1 y2-y1]);
    label=model.label;
    model=model.model;
    id=zeros(size(model,1),1);
    for i = 1 : size(model,1)
        id(i)=similarity(extractHOGFeatures(imresize(img1,[64 64])),model(i,:));
    end
    k=1;
    temp=[];
    while(k)
        [~,ind]=max(id);
        temp(end+1)=label(ind);
        id(ind)=0;
        k=k-1;
    end
    id=mode(temp);
end
function val = getCP(img,model,number)
    tmp=model.temp1_1;
    tmp=imresize(tmp,round([size(img,1)*0.0329 size(img,2)*0.0728]));
    corr=normxcorr2(rgb2gray(tmp),rgb2gray(img));
    x1=0.2177*size(img,2);
    x2=0.8523*size(img,2);
    y1=0.02*size(img,1);
    y2=0.15*size(img,1);
    [ypeak,xpeak] = find(corr(y1:y2,x1:x2)==max(max(corr(y1:y2,x1:x2))));
    yoffSet = ypeak-size(tmp,1);
    val=getNumber(img,xpeak,ypeak,yoffSet,x1,x2,y1,y2,1,number);
    if(val==-1)
        val=123;
    end
end
function val = getHP(img,model,number)
    tmp=model.temp1_2;
    tmp=imresize(tmp,round([size(img,1)*0.0228 size(img,2)*0.0522]));
    corr=normxcorr2(rgb2gray(tmp),rgb2gray(img));
    x1=0.25*size(img,2);
    x2=0.8*size(img,2);
    y1=0.4791*size(img,1);
    y2=0.5749*size(img,1);
    [ypeak,xpeak] = find(corr(y1:y2,x1:x2)==max(max(corr(y1:y2,x1:x2))));
    yoffSet = ypeak-size(tmp,1);
    val=getNumber(img,xpeak,ypeak,yoffSet,x1,x2,y1,y2,2,number);
    if(val==-1)
        val=26;
    elseif(length(int2str(val))<=2)
        return;
    else
        max_val=-1;
        val=int2str(val);
        n=length(val);
        if(mod(n,2)==0)
            m=n/2;
            v1=str2double(val(1:m));
            v2=str2double(val(m+1:n));
            if(v1<v2)
                v1=v2;
            end
            max_val=v1;
        else
            m1=floor(n/2);
            m2=ceil(n/2);
            v1=str2double(val(1:m1));
            v2=str2double(val(m1+1:n));
            v3=str2double(val(1:m2));
            v4=str2double(val(m2+1:n));
            if(v1<v2)
                v1=v2;
            end
            if(v1<v3)
                v1=v3;
            end
            if(v1<v4)
                v1=v4;
            end
            max_val=v1;
        end
        val=max_val;
    end
end
function val = getSD(img,model,number)
    tmp=model.temp1_4;
    tmp1=model.temp1_5;
    tmp=imresize(tmp,round([size(img,1)*0.0758 size(img,2)*0.4571]));
    tmp1=imresize(tmp1,round([size(img,1)*0.0898 size(img,2)*0.4303]));
    corr1=normxcorr2(rgb2gray(tmp),rgb2gray(img));
    corr2=normxcorr2(rgb2gray(tmp1),rgb2gray(img));
    x1=1;
    x2=0.8523*size(img,2);
    y1=0.7*size(img,1);
    y2=size(img,1);
    [ypeak1,xpeak1] = find(corr1(y1:y2,x1:x2)==max(max(corr1(y1:y2,x1:x2))));
    [ypeak2,xpeak2] = find(corr2(y1:y2,x1:x2)==max(max(corr2(y1:y2,x1:x2))));
    s1=2*abs(xpeak1-size(img,2)/2)/size(img,2);
    s2=2*abs(xpeak2-size(img,2)/2)/size(img,2);
    if(s1<s2)
        xpeak=xpeak1;
        ypeak=ypeak1;
        corr=corr1;
    else
        xpeak=xpeak2;
        ypeak=ypeak2;
        corr=corr2;
    end
    yoffSet = ypeak-size(tmp,1);
    val=getNumber(img,xpeak,ypeak,yoffSet,x1,x2,y1,y2,3,number);
    if(val==-1)
        val=600;
    else
        if(length(int2str(val))<3)
            val=val*10^2;
        end
    end
end
function val = getNumber(img,xpeak,ypeak,yoffSet,x1,x2,y1,y2,c,model)
    xf=0.0425;
    val=-1;
    max_val=-1;
    f=0;
    model=model.model;
    if(c==1)
        l=0.77;
        change=0.01;
    elseif(c==2 || c==3)
        l=0.99;
        change=-0.01;
    end
    while 1
        if((c==1 && l>=1) || (c==2 && round(l*100)<=50) || (c==3 && round(l*100)<=60))
            break;
        end
        img_1=imcrop(img,[x1 y1 x2-x1 y2-y1]);
        img1=im2bw(img_1,l);
        if(c==2 || c==3)
            img1(:)=img1(:)==0;
        end
        bw=bwconncomp(img1);
        numPixels = cellfun(@numel,bw.PixelIdxList);
        for j = 1 : size(numPixels,2)
            img2=zeros(size(img1,1),size(img1,2));
            img2(bw.PixelIdxList{j}) = 255;
            img2=im2uint8(img2);
            
            x11=1;
            x12=size(img2,2);
            y11=1;
            y12=size(img2,1);
            while sum(img2(:,x11))==0
                x11=x11+1;
            end
            while sum(img2(:,x12))==0
                x12=x12-1;
            end
            while sum(img2(y11,:))==0
                y11=y11+1;
            end
            while sum(img2(y12,:))==0
                y12=y12-1;
            end
            if(c==1)
                if(x11<xpeak-(xf*size(img1,2)) || y11 < yoffSet-(0.0556*size(img1,1)))
                    if(round(l*100)==99 && max_val==-1 && f<2)
                        xf=xpeak/size(img1,2);
                        l=0.76;
                        f=f+1;
                    end
                    continue;
                end
            elseif(c==2)
                if(y11 < yoffSet || y12 > ypeak)
                    if(round(l*100)==51 && max_val==-1 && f<1)
                        yoffSet=1;
                        ypeak=size(img2,1);
                        l=0.99;
                        f=f+1;
                    end
                    continue;
                end
            elseif(c==3)
                if(x12 < xpeak || x11 > 0.8*size(img1,2) || y11 < yoffSet || y12 > ypeak)
                    continue;
                end
            end
            img2=imresize(imcrop(img2,[x11 y11 x12-x11 y12-y11]),[32 32]);
            num=zeros(10,3);
            num(:)=-1;
            for k = 1 : 10
                if(c==1 || c==2 || c==3)
                    num(k,:)=[similarity(getFeat(img2),model(k,:)) x11 y11];
                else
                    num(k,:)=[similarity(extractHOGFeatures(img2),model(k,:)) x11 y11];
                end
            end
            [m,v]=max(num(:,1));
            if(m>0.85)
                if(val==-1)
                    val=v-1;
                else
                    val=(val*10)+(v-1);
                end
            end
         end
         if(val~=-1)
             if(max_val==-1)
                 max_val=val;
             elseif(max_val<val)
                 max_val=val;
             end
         end
         val=-1;
         l=l+change;
    end
    val=max_val;
end
function feat = getFeat(a)
    feat=[];
    for y = 1 : 4 : size(a,1)
        for x = 1 : 4 : size(a,2)
            feat(end+1)=round(sum(sum(imcrop(a,[x y 3 3])))/255);
        end
    end
end
function s = similarity(a,b)
    num=sum(a(1,:).*b(1,:));
    den=sqrt(sum(a(1,:).^2))*sqrt(sum(b(1,:).^2));
    s=num/den;
end