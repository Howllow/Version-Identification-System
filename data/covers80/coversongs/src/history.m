%-- Unknown date --%
plot([-500:500],r(3,:),[-500:500],fxc(3,:))
axis([0 1000 -.1 .8])
axis([-500 500 -.1 .8])
grid
subplot(412)
imgsc(chromnorm(Yb2(:,[1:630])))
imgsc(chromnorm(Yb2(:,20+[1:630])))
r = chromxcorr(chromnorm(chrompwr(Fes,.5)),chromnorm(chrompwr(Yb2(:,20+[1:630]),.5)),500);
for c = 1:12; fxc(c,:) = filter([1 -1],[1 -.9],r(c,:)-mean(r(c,:))); end
subplot(413)
imgsc([-500:500],[-5:6],chromrot(r,-5))
subplot(414)
plot([-500:500],r(3,:),[-500:500],fxc(3,:))
axis([-500 500 -.1 .8])
grid
print -depsc -adobecset elliott_smith-vs-glen_phillips.eps
%-- 1/25/07  9:52 AM --%
y = travel1(20,200,[-1],hanning(5)');
y = travel1(20,200,[-.25 -.5 -.25],[1]);
y = travel1(20,200,[-1],[1 0 0 0 0 0 0 0 0 0 1]);
subplot(211)
m = sintwavemov(2*pi/20,2*pi/40,-1,33,40);
movie(m,10)
wc = pi/4000;  % so f*wc is normalized angular frq of pole
r = 0.97;
r2 = r*r;
aah = conv(conv([1 -2*r*cos(500*wc) r2],[1 -2*r*cos(1500*wc) r2]), [1 -2*r*cos(2500*wc) r2]);
freqz(1,aah,512,8000)
pt = (rem(1:8000,80)==0);
vah = filter(1,aah,pt);
soundsc(vah)
aee = conv([1 -2*r*cos(250*wc) r2],[1 -2*r*cos(2300*wc) r2]);
freqz(1,aee,512,8000)
vee = filter(1,aee,pt);
soundsc(vee)
soundsc(vah)
aoo = conv([1 -2*r*cos(wc*330) r2],[1 -2*r*cos(wc*900) r2]);
freqz(1,aoo,512,8000)
voo = filter(1,aoo,pt);
soundsc(voo)
soundsc([vah,voo/8,vee])
x = pluck1a(40,10000);
soundsc(x)
subplot(211)
specgram(x,512,8000)
soundsc(x)
To get started, select MATLAB Help or Demos from the Help menu.
startup.m executing...
dpwestartup.m executing...
cwd is /Users/dpwe/docs/classes/e6820-2007-01/matlab
dpwestartup.m done
Warning: Usage of DATEVEC with empty date strings is not supported.
Results may change in future versions.
> In datevec at 95
In datenum at 89
In reporterrorlogs>getLogFiles at 212
In reporterrorlogs>getFilesToSend at 142
In reporterrorlogs at 26
In matlabrc at 245
>> y = travel1(20,200,[-1],hanning(5)');
Error in ==> travel1 at 58
pause
>> y = travel1(20,200,[-.25 -.5 -.25],[1]);
Error in ==> travel1 at 58
pause
>> y = travel1(20,200,[-1],[1 0 0 0 0 0 0 0 0 0 1]);
Error in ==> travel1 at 58
pause
>> subplot(211)
>> m = sintwavemov(2*pi/20,2*pi/40,-1,33,40);
>> movie(m,10)
>> wc = pi/4000;  % so f*wc is normalized angular frq of pole
r = 0.97;
r2 = r*r;
aah = conv(conv([1 -2*r*cos(500*wc) r2],[1 -2*r*cos(1500*wc) r2]), [1 -2*r*cos(2500*wc) r2]);
freqz(1,aah,512,8000)
pt = (rem(1:8000,80)==0);
vah = filter(1,aah,pt);
soundsc(vah)
>> aee = conv([1 -2*r*cos(250*wc) r2],[1 -2*r*cos(2300*wc) r2]);
freqz(1,aee,512,8000)
vee = filter(1,aee,pt);
soundsc(vee)
soundsc(vah)
>> aoo = conv([1 -2*r*cos(wc*330) r2],[1 -2*r*cos(wc*900) r2]);
freqz(1,aoo,512,8000)
voo = filter(1,aoo,pt);
soundsc(voo)
soundsc([vah,voo/8,vee])
>> soundsc([vah,voo/8,vee])
>> x = pluck1a(40,10000);
soundsc(x)
subplot(211)
specgram(x,512,8000)
>> specgram(x,512,8000)
>> soundsc(x)
>>
>> specgram(x,512,8000)
>> soundsc(x)
>>
??? Undefined function or method 'To' for input arguments of type 'char'.
>>
??? >> specgram(x,512,8000)
|
Error: Unexpected MATLAB operator.
>>
??? >>
|
Error: Unexpected MATLAB operator.
>>
??? >>
|
Error: Unexpected MATLAB operator.
>>
??? >>
|
Error: Unexpected MATLAB operator.
>>
y = travel1(20,200,[-1],hanning(5)');
y = travel1(20,200,[-.25 -.5 -.25],[1]);
y = travel1(20,200,[-1],[1 0 0 0 0 0 0 0 0 0 1]);
m = sintwavemov(2*pi/20,2*pi/40,-1,33,40);
subplot(211)
m = sintwavemov(2*pi/20,2*pi/40,-1,33,40);
movie(m,10)
help sintwavemov
m = sintwavemov(2*pi/20,2*pi/40,-.9,33,40);
m = sintwavemov(2*pi/20,2*pi/40,-.6,33,40);
movie(m,10)
soundsc([vah,voo/8,vee])
freqz(1,aoo,512,8000)
freqz(1,aee,512,8000)
%-- 1/25/07  5:47 PM --%
help PBmodel
PBmodel
1000
10
BManimation
PBmodel
10000
10
BManimation
PBmodel
10000
10
BManimation
PBmodel
1000
10
BManimation
%-- 1/30/07  9:54 AM --%
FeatureExtractionPopen
FeaturesExtractionPopen
which PAaddpath.m
FeaturesExtractionPopen
which mp3read
FeaturesExtractionPopen
which mp3read
FeaturesExtractionPopen
d = mp3dir('../Input/AudioData');
d
size(d)
d(1)
d{1}
help fullfile
d = mp3dir('../Input/AudioData');
d{1}
dd = d{1}
isstring(d)
iscell(d)
isstring(dd)
iscell(d)
cell(dd)
cell(dd,1)
dd
help cell
ddc = cell(1);
size(ddc)
ddc = dd;
ddc = cell(1);
ddc{1} = dd;
ddc
ddc{1}
FeaturesExtractionPopen
D = dir('../Input/AudioData/*.mp3');
size(D)
D(1)
d = mp3dir('../Input/AudioData');
d{1}
FeaturesExtractionPopen
rehash
FeaturesExtractionPopen
help fileparts
FeaturesExtractionPopen
FeaturesExtractionPopen('/Users/dpwe/Documents/audiodiary/2007-01-25-1330.mp3');
FeaturesExtractionPopen
FeaturesExtractionPopen('/Users/dpwe/Documents/audiodiary/2007-01-25-1330.mp3');
FeaturesExtractionPopen('/Users/dpwe/Documents/audiodiary/2007-01-03-1833.mp3');
cd ..
mkimgs
ls
makeimgs
pwd
en = load('Output/Features/2007-01-25-1330_AvgLEnergy.mat');
eh = load('Output/Features/2007-01-25-1330_AvgEntropy.mat');
nd = load('Output/Features/2007-01-25-1330_EnergyDev.mat');
nd = load('Output/Features/2007-01-25-1330_LEnergyDev.mat');
nh = load('Output/Features/2007-01-25-1330_EntropyDev.mat');
subplot(411)
imgsc(en)
en
imgsc(en.AvgLEnergy)
subplot(412)
imgsc(nd.LEnergyDev)
subplot(413)
imgsc(eh.AvgEntropy)
subplot(414)
imgsc(hd.EntropyDev)
hd
imgsc(nh.EntropyDev)
rgb = imghsv(nh.EntropyDev', eh.AvgEntropy', en.AvgLEnergy');
imwrite(rgb,'tmp.jpg','jpeg');
pwd
makeimgs
a = '2006-07-19-1527_AvgLEnergy.mat
a = '2006-07-19-1527_AvgLEnergy.mat';
find(a == '_')
makeimgs
cd Code
FeaturesExtractionPopen('2006-12-05-1036.mp3');
FeaturesExtractionPopen('../../Input/AudioData/2006-12-05-1036.mp3');
pwd
ls ../Input/AudioData/
dir('../../Input/AudioData/2006-12-05-1036.mp3')
pwd
dir('../Input/AudioData/2006-12-05-1036.mp3')
FeaturesExtractionPopen('../Input/AudioData/2006-12-05-1036.mp3');
makeimgs
cd ..
makeimgs
help imwrite
help image
help imagergb
help imview
help imshow
makeimgs
load ../Output/Features/2006-12-05-1036_AvgLEnergy
load ../Output/Features/2006-12-05-1036_AvgLEnergy.mat
pwd
load Output/Features/2006-12-05-1036_AvgLEnergy.mat
subplot(311)
imgsc(AvgLEnergy
imgsc(AvgLEnergy)
subplot(312)
load Output/Features/2006-12-05-1036_AvgEntropy.mat
imgsc(AvgEntropy)
subplot(313)
load Output/Features/2006-12-05-1036_EntropyDev.mat
imgsc(EntropyDev)
help bicseg
[T,B] = bicseg(AvgEntropy);
help pca
[v,s,AEp] = pca(AvgLEnergy',5);
subplot(313)
imgsc(AEp')
[v,s,AHp] = pca(AvgEntropy',5);
imgsc(AHp')
[T,B] = bicseg([AEp([1 2],:)';AHp([1 2],:)']);
[T,B] = bicseg([AEp([1 2],:)';AHp([1 2],:)]);
size(AEp)
[T,B] = bicseg([AEp(:,[1 2])';AHp(:,[1 2])']);
T
[T,B] = bicseg([AEp(:,[1 2])';AHp(:,[1 2])'],5,1,.5);
B
T
[T,B] = bicseg([AEp(:,[1 2])';AHp(:,[1 2])'],5,1,5);
[T,B] = bicseg([AEp(:,[1 2])';AHp(:,[1 2])'],5,1,3);
B
T
hold on;
plot([T;T],[0.5 5.5],'-w')
[T,B] = bicseg([AEp(:,[1 2])';AHp(:,[1 2])'],5,1,5);
subplot(311)
plot([T;T],[0.5 21.5],'-w')
imgsc(AEp')
imgsc(AvgLEnergy)
hold on;plot([T;T],[0.5 21.5],'-w'); hold off
load Output/Features/2006-12-05-1036_EntropyDev.mat
load Output/Features/2006-12-05-1036_AvgLEnergy.mat
load Output/Features/2006-12-05-1036_EntropyDev.mat
[v,s,AEp] = pca(AvgLEnergy',5);
[v,s,AHp] = pca(AvgEntropy',5);
[T,B] = bicseg([AEp(:,[1 2])';AHp(:,[1 2])'],5,1,5);
subplot(311)
imgsc(AvgLEnergy)
hold on;plot([T;T],[0.5 21.5],'-w'); hold off
B
T
size(AEp)
%-- 1/31/07 10:45 PM --%
[d,sr] = mp3read('/Users/dpwe/projects/meapsoft/old/MEAPsoft-dev/mums-piano.mp3
[d,sr] = mp3read('/Users/dpwe/projects/meapsoft/old/MEAPsoft-dev/mums-piano.mp3',[1 88000],1,2);
soundsc(d,sr)
plot(d)
plotspec(d,16384);
help plotspec
plotspec(d,sr,16384,1000);
hold on; plot([1;1]*[1:40]*27.5,[-100 -30],'-g')
axis([0 500 -100 -50])
hold of;
hold off;
plot([1;1]*[1:40]*27.5,[-100 -50],'-g')
hold on;
plotspec(d,sr,32768,500);
axis([0 500 -100 -50])
subplot(211)
hold of
hold off
plot([1;1]*[1:40]*27.5,[-100 -40],'-g')
hold on
plotspec(d,sr,32768,500);
hold off
axis([0 500 -100 -40])
title('MUMS Grand Piano - A0 (27.5 Hz) - 32768 pts @ 22 kHz')
print -djpeg pianoA0.jpg
32768/sr
%-- 2/1/07  9:45 AM --%
load fmtO.txt
load fmtU.txt
load fmtA.txt
% Build 2-dimensional data set [F1, F2]
dat = [fmtO(:,[1 2]);fmtU(:,[1 2]);fmtA(:,[1 2])];
size(dat)
%ans =
%   150     2
% 50 examples of each
% Build target outputs.  One unit for O, one for U, one for A
oo = ones(50,1); zz = zeros(50,1);
tgt = [oo,zz,zz;zz,oo,zz;zz,zz,oo];
size(tgt)
%ans =
%   150     2
% Calculate normalization, so input units don't saturate
nrm = [mean(dat);std(dat)];
% Train net, 5 HUs, 50 epochs
[wh,wo,es] = nntrain(dat,nrm,tgt,0.1,50,5);
%Iteration=1 MSError =0.88821
%Iteration=2 MSError =0.64701
%...
%Iteration=49 MSError =0.22906
%Iteration=50 MSError =0.22845
% Continue training with reduced learning rate
[wh,wo,es] = nntrain(dat,nrm,tgt,0.05,10,wh,wo);
%Iteration=1 MSError =0.22299
%...
%Iteration=10 MSError =0.21944
[wh,wo,es] = nntrain(dat,nrm,tgt,0.025,10,wh,wo);
%Iteration=1 MSError =0.21249
%...
%Iteration=10 MSError =0.21339
% Look at the performance on the training data.  Should flip after 50
nno = nnfwd(dat,nrm,wh,wo);
plot(nno)
plot(fmtU(:,1),fmtU(:,2),'.r',fmtO(:,1),fmtO(:,2),'.b',fmtA(:,1),fmtA(:,2),'.g');
% Sample entire 2D surface of NN outputs
[nnO,xx,yy] = nngridsamp(wh,wo,nrm,[200 1100 600 1600],60,1);
[nnU,xx,yy] = nngridsamp(wh,wo,nrm,[200 1100 600 1600],60,2);
[nnA,xx,yy] = nngridsamp(wh,wo,nrm,[200 1100 600 1600],60,3);
% Use contour to plot where their difference crosses zero
hold on
contour(xx,yy,nnO-max(nnU,nnA),[0 0])
contour(xx,yy,nnA-max(nnU,nnO),[0 0])
% Pretty good decision boundary.  You can see the 'wrong' points
% Add the actual surfaces defined by the outputs:
surf(xx,yy,nnO)
surf(xx,yy,nnU)
surf(xx,yy,nnA)
hold off
figure 2
figure(2)
plot(fmtU(:,1),fmtU(:,2),'.r',fmtO(:,1),fmtO(:,2),'.b',fmtA(:,1),fmtA(:,2),'.g');
hold on
contour(xx,yy,nnA-max(nnU,nnO),[0 0])
contour(xx,yy,nnO-max(nnU,nnA),[0 0])
figure(3)
hold off;
[gmm,gmv,gmc]=gmmest(dat,3,[],[],50,1);
[gmm,gmv,gmc]=gmmest(dat,5,[],[],50,1);
[d,sr] = mp3read('/Users/dpwe/Desktop/mums-piano-E1.mp3');
size(d)
sr
plotspec(d,sr,2^17,500);
hold on;
plot([1;1]*[1:12]*44.2,[-100 -40],'-g')
hold off
[d,sr] = mp3read('/Users/dpwe/Desktop/mums-piano-E#1.mp3',0,1,2);
size(d)
plotspec(d,sr,2^17,500);
hold on; plot([1;1]*[1:12]*44.2,[-100 -40],'-g'); hold off
plotspec(d,sr,2^17,1000);
hold on; plot([1;1]*[1:20]*44.2,[-100 -40],'-g'); hold off
[d,sr] = mp3read('/Users/dpwe/Desktop/mums-piano-E1.mp3',0,1,2);
plotspec(d,sr,2^16,1000);
hold on; plot([1;1]*[1:20]*41.2,[-100 -40],'-g'); hold off
hold on; plot([1;1]*[1:24]*41.2,[-100 -40],'-g'); hold off
[db,sr] = wavread('/Users/dpwe/Desktop/mums-dbl-bass-E1.wav');
size(db)
subplot(211)
plotspec(d,sr,2^16,1000);
hold on; plot([1;1]*[1:24]*41.2,[-100 -40],'-g'); hold off
title('MUMS Grand Piano - E1 - 3s window');
2^16/sr
subplot(212)
plotspec(db,sr,2^16,1000);
soundsc(db,sr)
soundsc(d,sr)
[db,sr] = ReadSound('/Users/dpwe/Desktop/mums-dbl-bass-E1.aiff');
[db,sr] = wavread('/Users/dpwe/Desktop/mums-dbl-bass-E1.wav');
soundsc(db,sr)
soundsc(d,sr)
plotspec(db,sr,2^16,1000);
hold on; plot([1;1]*[1:24]*41.2,[-100 -40],'-g'); hold off
hold on; plot([1;1]*[1:24]*41.2,[-100 -20],'-g'); hold off
title('MUMS Plucked Double Bass - E1 - 3s window');
plot([1;1]*[1:24]*41.2,[-100 -20],'-g')
hold on
plotspec(db,sr,2^16,1000);
hold of
hold off
title('MUMS Plucked Double Bass - E1 - 3s window');
subplot(211)
plot([1;1]*[1:24]*41.2,[-100 -20],'-g')
plotspec(d,sr,2^16,1000);
plot([1;1]*[1:24]*41.2,[-100 -20],'-g')
hold on
plotspec(d,sr,2^16,1000);
hold off
title('MUMS Grand Piano - E1 - 3s window');
orient landscape
print -djpeg piano-vs-bass-E1.jpg
%-- 2/4/07 10:38 AM --%
publish popenr_demo
norm([3])
norm([3 3])
norm([3 3 3])
norm([3 3 3])/sqrt(3)
publish popenr_demo
help publish
publish popenr_demo
[d,sr] = mp3read('piano.mp3');
size(d)
mp3write(d,sr,'piano2.mp3');
[d2,sr ] = mp3read('piano2.mp3');
size(d2)
size(d)
rms(d - d2(1:208559,:))
rms(d - d2(1:207407,:))
norm(d - d2(1:207407,:))
norm(d - d2(1+[1:207407],:))
norm(d - d2(2+[1:207407],:))
norm(d - d2(200+[1:207407],:))
norm(d - d2([1:207407],:))
norm(d(2:end) - d2([1:207406],:))
norm(d(2:end,:) - d2([1:207406],:))
norm(d(2:end,:) - d2(1+[1:207406],:))
norm(d(2:end,:) - d2(2+[1:207406],:))
help mp3read
help mp3write
norm(d)
diff = (d(2:end,:) - d2(2+[1:207406],:));
plot(diff(1:1000,1))
subplot(211)
plot(diff(1:1000,1))
subplot(212)
plot(d(1:1000,1))
plot(d(10000+[1:1000],1))
subplot(211)
plot(diff(10000+[1:1000],1))
mean(sum(d - d2(1:207406,:)))
mean(sum(d - d2(1:207407,:)))
size(d)
mean(sum((d - d2(1:207407,:)).^2))
sqrt(mean(sum((d - d2(1:207407,:)).^2)))
sqrt(mean(sum((d).^2)))
mp3write(d2,sr,'piano3.mp3');
[d3,sr ] = mp3read('piano3.mp3');
sqrt(mean(sum((d2 - d3).^2)))
size(d2)
size(d3)
size(d2(1:207407,:)-d3(1:207407,:))
sqrt(mean(sum((d2(1:207407,:)-d3(1:207407,:)).^2)))
sqrt(mean(sum((d2(2:207407,:)-d3(1:207406,:)).^2)))
sqrt(mean(sum((d3(2:207407,:)-d2(1:207406,:)).^2)))
sqrt(mean(sum((d3(2:207407,:)).^2)))
subplot(211)
plot(1:1000,d(10000+[1:1000],1),1:1000,d2(10000+[1:1000],1))
plot(diff)
subplot(212)
plot(d)
d2 = d2(1:207407,:);
sqrt(mean(sum((d(:)-d2(:)).^2)))
sqrt(mean(sum((d(1:410000)-d2(1:410000)).^2)))
sqrt(mean(sum((d(1:410000)-d2(2:410001)).^2)))
sqrt(mean(sum((d(3:410002)-d2(2:410001)).^2)))
mean(sum(d(:).^2))
mean(sum(d2(:).^2))
aa = mean(sum(d2(:).^2))/mean(sum(d2(:).^2));
aa
aa = mean(sum(d2(:).^2))/mean(sum(d(:).^2));
aa
mean(sum((d - d2).^2))
sqrt(mean(sum((d - d2).^2)))
sqrt(mean(sum((d*aa - d2).^2)))
sqrt(mean(sum((d - aa*d2).^2)))
sqrt(mean(sum((d*aa - d2).^2)))
sqrt(mean(sum((d*.9 - d2).^2)))
sqrt(mean(sum((d*.9005 - d2).^2)))
sqrt(mean(sum((d*.9006 - d2).^2)))
sqrt(mean(sum((d*.9007 - d2).^2)))
sqrt(mean(sum((d*.901 - d2).^2)))
sqrt(mean(sum((d*.91 - d2).^2)))
sqrt(mean(sum((d*.92 - d2).^2)))
aa = sqrt(mean(sum(d2(:).^2))/mean(sum(d(:).^2)));
aa
sqrt(mean(sum((d*aa - d2).^2)))
sqrt(mean(sum((d*.9489 - d2).^2)))
sqrt(mean(sum((d*.949 - d2).^2)))
sqrt(mean(sum((d*.9488 - d2).^2)))
sqrt(mean(sum((d*.9487 - d2).^2)))
sqrt(mean(sum((d*.9486 - d2).^2)))
sqrt(mean(sum((d*.9482 - d2).^2)))
sqrt(mean(sum((d*.9481 - d2).^2)))
sqrt(mean(sum((d*.948 - d2).^2)))
sqrt(mean(sum((d*.947 - d2).^2)))
sqrt(mean(sum((d*.946 - d2).^2)))
sqrt(mean(sum((d*.945 - d2).^2)))
sqrt(mean(sum((d*.947 - d2).^2)))
sqrt(mean(sum((d(:)*.947 - d2).^2)))
sqrt(mean(sum((d(:)*.947 - d2(:)).^2)))
sqrt(mean(sum((d(:)*.948 - d2(:)).^2)))
sqrt(mean(sum((d(:)*.946 - d2(:)).^2)))
sqrt(mean(sum((d(:)*.947 - d2(:)).^2)))
sqrt(mean(sum((d(1:410000)*.947 - d2(1:410000)).^2)))
sqrt(mean(sum((d(1:410000)*.947 - d2(2+[1:410000])).^2)))
sqrt(mean(sum((d(2+[1:410000])*.947 - d2(2+[1:410000])).^2)))
sqrt(mean(sum((d(2+[1:410000])*.947 - d2(0+[1:410000])).^2)))
norm(d - d2(1:207407,:))
norm(d*.947 - d2(1:207407,:))
1/.947
norm(d - 1.056*d2(1:207407,:))
norm(d - 1.055*d2(1:207407,:))
norm(d - 1.054*d2(1:207407,:))
norm(d - 1.053*d2(1:207407,:))
norm(d - 1.05*d2(1:207407,:))
norm(d - 1.052*d2(1:207407,:))
norm(d - 1.051*d2(1:207407,:))
norm(d - 1.052*d2(1:207407,:))
sysinfo
machinfo
arch
machtype
help
computer
isunix
ispc
help computer
help switch
help computer
help fileparts
help fullfile
fullfile('/usr', 'tmp', '')
which mp3read
addpath('/Users/dpwe/matlab/columbiafns');
which mp3read
[d3,sr ] = mp3read('piano3.mp3');
help mkdir
[d3,sr ] = mp3read('piano3.mp3');
soundsc(d3,sr)
help mkdir
help fileattrib
fileattrib('tmp')
help exists
help exist
exist('tmp','file')
exist('/tmp','file')
help wavread
help ischar
ischar('native')
help tolower
help lowercase
help lower
ls
wavwrite(d,sr,'piano.wav');
[dw,sr] = wavread('piano.wav','native');
dw(1:10)
whos
help int16
[dn,sr] = mp3read('piano.mp3','native');
[dn,sr] = mp3read('piano.mp3');
dn(1:10)
[dn,sr] = mp3read('piano.mp3',);
[dn,sr] = mp3read('piano.mp3','native');
dn(1:10)
[dn,sr] = mp3read('piano.mp3','native');
dn(1:10)
[dw,sr] = wavread('piano.wav','native');
dn(1:10)
[dw,sr] = wavread('piano.wav',100);
[dn,sr] = mp3read('piano.mp3',100);
size(dw)
size(dn)
dw(1:10)
dN(1:10)
dn(1:10)
[dn,sr] = mp3read('piano.mp3',[1000 2000]);
[dw,sr] = wavread('piano.wav',[1000 2000]);
size(dn)
size(dw)
dn(1:100)
dn(1:10)
dw(1:10)
[dw,sr] = wavread('piano.wav',[1000 2000],'double');
[dn,sr] = mp3read('piano.mp3',[1000 2000],'double');
dn(1:10)
dw(1:10)
[dw,sr] = wavread('piano.wav',[1000 2000],'native');
[dn,sr] = mp3read('piano.mp3',[1000 2000],'native');
dn(1:10)
dw(1:10)
[dn,sr] = mp3read('piano.mp3',[1000 2000],'native');
[dn,sr] = mp3read('piano.mp3',[1000 2000],'nativee');
help computeMetric
help computeMetrics
addpath('/Users/dpwe/Desktop/fxtoolbox')
pwd
cd ..
pwd
codemetrics
codemetrics('/Users/dpwe/docs/classes/e4810-2006-09/matlab/lagrangepoly')
codemetrics('/Users/dpwe/matlab/columbiafns/mp3readwrite')
help exist
exist('popenw')
exist('popenw','builtin')
which popenw
which popenww
help which
which popenww
a = which('popenw')
a = which('popenww')
length(which('popenww'))
length(which('popenw'))
codemetrics('/Users/dpwe/matlab/columbiafns/mp3readwrite')
help wavread
siz = mp3read('piano.mp3', 'size')
addpath('/Users/dpwe/matlab/columbiafns/mp3readwrite');
siz = mp3read('piano.mp3', 'size')
pwd
ls
cd popenmatlab
ls
siz = mp3read('piano.mp3', 'size')
siz = mp3read('piano', 'size')
[p,n,e] = fileparts('piano')
siz = mp3read('piano', 'size')
siz = mp3read('piano.mp3', 'size')
[dn,sr] = mp3read('piano.mp3',[1000 2000],'nativee');
[dn,sr] = mp3read('piano.mp3',[1000 2000],'native');
[dn,sr] = mp3read('piano',[1000 2000]);
[dw,sr] = wavread('piano',[1000 2000]);
size(dn)
dn(1:10)
size(dw)
dw(1:10)
codemetrics('/Users/dpwe/matlab/columbiafns/mp3readwrite')
[dn,sr] = mp3read('piano',[1000 2000]);
dn(1:10)
siz = mp3read('piano.mp3', 'size')
codemetrics('/Users/dpwe/matlab/columbiafns/mp3readwrite')
help mp3read
help wavread
help mp3read
help wavwrite
1152/44100
wavwrite(d2,sr,'piano3');
;s
ls
mp3write(d2,sr,'piano32');
codemetrics('/Users/dpwe/matlab/columbiafns/mp3readwrite')
mp3write(d2,sr,16,'piano32');
size(d2)
mp3write(d2,sr,'piano32');
[dn,sr] = mp3read('piano',[1000 2000]);
which mp3read
[dn,sr] = mp3read('piano',[1000 2000]);
mp3write(dn,sr,'piano32');
[dn2,sr] = mp3read('piano32');
[dn2,sr] = mp3read('piano');
publish demo_mp3readwrite
demo_mp3readwrite
publish demo_mp3readwrite
size(d2)
disp(['SNR is ',num2str(10*log10(sum(d(:).^2)/sum(ddiff(:).^2))),' dB']);
ddiff = d - d2;
disp(['SNR is ',num2str(10*log10(sum(d(:).^2)/sum(ddiff(:).^2))),' dB']);
ddiff = d - 1.05*d2;
disp(['SNR is ',num2str(10*log10(sum(d(:).^2)/sum(ddiff(:).^2))),' dB']);
ddiff = d - 1.051*d2;
disp(['SNR is ',num2str(10*log10(sum(d(:).^2)/sum(ddiff(:).^2))),' dB']);
ddiff = d - 1.049*d2;
disp(['SNR is ',num2str(10*log10(sum(d(:).^2)/sum(ddiff(:).^2))),' dB']);
ddiff = d - 1.048*d2;
disp(['SNR is ',num2str(10*log10(sum(d(:).^2)/sum(ddiff(:).^2))),' dB']);
ddiff = d - 1.055*d2;
disp(['SNR is ',num2str(10*log10(sum(d(:).^2)/sum(ddiff(:).^2))),' dB']);
ddiff = d - 1.052*d2;
disp(['SNR is ',num2str(10*log10(sum(d(:).^2)/sum(ddiff(:).^2))),' dB']);
publish demo_mp3readwrite
pwd
publish demo_mp3readwrite
addpath('/Users/dpwe/matlab/columbiafns/my mp3readwrite');
which mp3read
[dn2,sr] = mp3read('piano');
mp3write(dn,sr,'piano32');
which mp3write
mp3write(dn,sr,'piano32');
which mp3write
rehash
which mp3write
mp3write(dn,sr,'piano32');
pwd
mp3write(d,sr,'piano32');
help env
help general
help getenv
fullfile('','path','file')
fullfile('path','file')
[dn2,sr] = mp3read('piano');
ls
getenv TMPDIR
getenv('TMPDIR')
fullfile('usr','tmp')
help fullfile
[dn2,sr] = mp3read('piano');
ff = fullfile('/usr','tmp')
exist(ff,'file')
ls /usr/tmp
ls /usr
pwd
[dn2,sr] = mp3read('piano');
ls /tmp
publish demo_mp3readwrite
rehash
which demo_mp3readwrite
publish demo_mp3readwrite
addpath('/Users/dpwe/matlab/columbiafns/mp3readwrite');
publish demo_mp3readwrite
%-- 2/5/07  3:48 PM --%
which fexist
help fexist
ls
help exist
help fexist
help fexists
ls
help calclistftrs
less calclistftrs
qlist = calclistftrs('1.wav','2.wav');
qlist = calclistftrs('../tmp.list');
ls ../../OWL/testwav/Abracadabra/
qlist = calclistftrs('1.wav','2.wav');
%-- 2/5/07  7:21 PM --%
pwd
ls *mp3
type list.txt
qlist = calclistftrs('list.txt');
R = coverTestLists(qlist)
qlist = calclistftrs('list.txt');
R = coverTestLists(qlist)
qlist = calclistftrs('list.txt');
60/(106*.004)
qlist = calclistftrs('list.txt');
[d,sr] = mp3read('river_green.mp3',0,1,2);
rg = load('river_green');
rg
help mkblisp
help mkblips
db = mkblips(fg.bts,sr,length(d));
db = mkblips(rg.bts,sr,length(d));
soundsc(d+db,sr)
rg.d = d;
rg.sr = sr;
[d,sr] = mp3read('river_lennox.mp3',0,1,2);
rg.db = db;
rl = load('river_lennox');
rl.db = mkblips(rl.bts,sr,length(d));
rl.d = d;
rl.sr = sr;
soundsc(rl.d(1:20*sr)+rl.db(1:20*sr),sr)
