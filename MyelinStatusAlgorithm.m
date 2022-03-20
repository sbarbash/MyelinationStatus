% The following script calculates myelin index for an image.
% It corrects image intensity, detects oligodendrocyte cells, finds
% myelinating areas and calculates the ratio between myelinating areas
% and total oligodendrocyte area

% neuron_P is neuron image
% oligo_P is oligodendrocyte image

% image correction
I_neuron_P=imadjust(neuron_P,stretchlim(neuron_P),[]);
I_oligo_P=imadjust(oligo_P,stretchlim(oligo_P),[]);

% detecting oligo cells
reIdxInC_P=OligoCells(I_oligo_P,30); % Gaussian filter size determined manualy

% finding myelinating parts
[Lmyelin_P,Lolig_P,reIdxInC_P,Leak_P]=OligoProj(I_oligo_P,I_neuron_P,reIdxInC_P);

%calculate ratio
OligArea=regionprops(Lolig_P,'area');
MyelinArea=regionprops(Lmyelin_P,'area');
OligArea=[OligArea.Area]';
MyelinArea=[MyelinArea.Area]';
MyelinIndex=MyelinArea./OligArea; % myelination index for each cell


% auxilary functions
function [reIdxInC]=OligoCells(oligo,GaussW)
%OligoCells recieves oligodendrocyte image and a Gaussian filter size GaussW
% and returns reIdxInC: affiliation of each pixel in the image to an
% individual cell.

StrCom=[0.1 0.99]; % fraction of the image to saturate at low and high values
SizeLim=[200 4000]; % lower and upper limit of cell

% correcting leak from neuronal channel
oligo=imadjust(oligo,stretchlim(oligo,StrCom),[]);
oligo_blur_comp=imcomplement(imgaussfilt(oligo,80,'FilterDomain','spatial')); 
% Gaussian filter size 80 determined manualy
oligo=imsubtract(oligo,oligo_blur_comp.*0.4); %0.4 is correction gain, can be changed manualy
oligo=imadjust(oligo,stretchlim(oligo,StrCom),[]);

% large blur to only see cells
oligo_blur=imgaussfilt(oligo, GaussW,'FilterDomain','spatial');

%Global image threshold using Otsu's method
BW=oligo_blur>multithresh(oligo_blur);

% split the area accoridng to sectors matching objects in L.
[B L]=bwboundaries(BW,'noholes');

% for sectors go over all pixels and put the number of the closest object
reIdxInC=SplitSectors(B,L);
end


function [Lmyelin, Lolig, reIdxInC, Leak]=OligoProj(oligo,neuron,reIdxInC)
%OligoProj recives two channels; neuron and oligo and reIdxInC; pixel 
%affiliation with each cell, and returns boundaries as labeled matrix L 
% for the myelinated parts and the entire oligo. It also changes reIdxInC
% and outpus the new one, and measures Leak; leakage between neuron and oligo. 
Lmyelin=zeros(size(oligo));
Lolig=zeros(size(oligo));
New_reIdxInC=zeros(size(oligo));

oligo=imadjust(oligo,stretchlim(oligo),[]);
neuron=imadjust(neuron,stretchlim(neuron),[]);

% filter neuron for oligo leak
neuron_B=imgaussfilt(neuron, 35,'FilterDomain','spatial');
neuron=imsubtract(neuron,neuron_B.*0.8);
% filter size 35 and correction gain 0.8 determined manualy
neuron=imadjust(neuron,stretchlim(neuron),[]);

% filter oligo for neuron leak
globalPicks=imgaussfilt(oligo,80,'FilterDomain','spatial');
globalPicksNorm=adapthisteq(globalPicks,'ClipLimit',0.02);
oligo_blur_comp=imcomplement(globalPicksNorm);
oligo=imsubtract(oligo,oligo_blur_comp.*0.4);
% filter size 80, correction gain 0.8 and ClipLimit 0.02 determined manualy
oligo=imadjust(oligo,stretchlim(oligo),[]);

% bluring for smoother images
oligo_blur_all=imgaussfilt(oligo, 2.5,'FilterDomain','spatial');
neuron_blur_all=imgaussfilt(neuron, 2.5,'FilterDomain','spatial');
% filter size 2.5 determined manualy

% now start per cell
count=1;
for cc=1:max(reIdxInC(:))
    % specific cell region
    CellCoor=ones(size(reIdxInC));
    CellCoor(reIdxInC~=cc)=nan;
    oligo_blur=double(oligo_blur_all).*(CellCoor);
    neuron_blur=double(neuron_blur_all).*(CellCoor);
    % substracting background
    [a b]=hist(oligo_blur(:),5000);
    [aa bb]=max(a);
    if bb+300<=5000
        mO=oligo_blur>b(bb+70);
    else
        mO=oligo_blur>b(5000);
    end
    mO=bwareaopen(mO,100);
    if sum(mO(:))<2000
        continue
    end
    
    [a b]=hist(neuron_blur(:),5000);
    [aa bb]=max(a);
    if bb+300<=5000
        mN=neuron_blur>(b(bb+300));
    else
        mN=neuron_blur>(b(5000));
    end
    mN=bwareaopen(mN,100);
    
    a=mO.*double(oligo); a=a(:);
    b=mO.*double(neuron);b=b(:);
    a(a==max(a))=0; b(b==max(b))=0;
    ind=a~=0&b~=0;
    preCorC=corrcoef(a(ind),b(ind));
    CorC(cc)=preCorC(1,2);
    if abs(CorC(cc))>0.5
        continue
    end
    Leak(count)=CorC(cc);
    Lolig=Lolig+mO*count;
    Lmyelin=Lmyelin+mO.*mN*count;
    New_reIdxInC(reIdxInC==cc)=count;
    count=count+1;
end
reIdxInC=New_reIdxInC;
end

function reIdxInC=SplitSectors(B,L)
%SplitSectors recieve labeld matrix L with cells marked as indices and B; 
%a boundaries array and returns reIdxInC; pixel affiliation with each cell
%i.e. sectors.

s=size(L);
%organize Mdl
MdlWithC=[B{1} ones(length(B{1}),1)];
for ii=2:length(B)
    MdlWithC=[MdlWithC;[B{ii} ii*ones(length(B{ii}),1)]];
end
Mdl=MdlWithC(:,1:2);
%organize Y
M=ones(s);
[a b]=find(M==1);
Y=[a b];

Idx=knnsearch(Mdl,Y); % Mdl is cells Y is quary pixel
IdxInC=MdlWithC(Idx,3);
reIdxInC=reshape(IdxInC,s);
end