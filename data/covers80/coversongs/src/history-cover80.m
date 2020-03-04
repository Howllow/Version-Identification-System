models(1)
exist('md
help exist
exist('models')
train_all
test_labs
test_labs(10)
test_labs{10}
ulabs
strcmp(ulabs,'u2')
find(strcmp(ulabs,'u2'))
train_all
strcmp(ulabs, test_labs{file})
strcmp(ulabs,'u2')
test_labs(file)
file
train_all
size(indx)
size(gt)
indx(1:10)
gt(1:10)
imgsc(gt)
imgsc(lhood)
clear models
train_all
indx
indx == gt
mean(indx==gt)
size(mean)
clear mean
mean(indx==gt)
imgsc(lhood)
ulabs(10)
cm = 0*lhood;
for i=1:451; cm(indx(i),i) = 1; end
imgsc(cm)
gtm = 0*lhood;
for i=1:451; cm(gt(i),i) = 1; end
imgsc(gtm)
gt(i)
for i=1:451; gtm(gt(i),i) = 1; end
imgsc(gtm)
cm = 0*lhood;
for i=1:451; cm(indx(i),i) = 1; end
imgsc(gtm*cm')
subplot(121)
imgsc(gtm*cm')
subplot(122)
imgsc(lhood)
subplot(121); imgsc(cm*gtm')
ulabs(2)
diag(cm*gtm')
diag(cm*gtm')'
length(ulabs)
mean(indx==gt)
train_all
1.18
1/18
clear models
train_all
unames
whos
ulabs
train_all
clear models
train_all
sum(confusm)
sum(confusm')
sum(strcmp(test_labs,ulabs{1})
sum(strcmp(test_labs,ulabs{1}))
sum(strcmp(test_labs,ulabs{2}))
sum(strcmp(test_labs,ulabs{3}))
sum(strcmp(test_labs,ulabs{4}))
sum(diag(confusm))/sum(sum(confusm))
tic; [er,cfm,lhd] = train_all(); toc
tic; [er,cfm,lhd] = do_expt(); toc
er
cfm
diag(cfm)
diag(cfm)'
sum(cfm)
sum(cfm')
diag(cfm)'./sum(cfm)
diag(cfm)'./sum(cfm')
printf('%d ',diag(cfm))
fprintf(stdout,'%d ',diag(cfm))
fprintf(1,'%d ',diag(cfm))
fprintf(1,'%d ',diag(cfm)'./sum(cfm'))
%-- 4/04/07  9:29 PM --%
vary_params
ls
vary_parms
which gmm
exist('gmm')
addpath('/homes/drspeech/share/lib/matlab/netlab')
exist('gmm')
quit
%-- 4/04/07  9:39 PM --%
vary_parms
whos
quit
%-- 4/19/07  4:08 AM --%
vary_parms
>> tic; [a3,c3,l3,m3] = do_expt_chroma('tracks-train.txt','tracks-test.txt',64,1000,4,1,1); toc
tic; [a3,c3,l3,m3] = do_expt_chroma('tracks-train.txt','tracks-test.txt',16,1000,4,0,1); toc
tic; [a3,c3,l3,m3] = do_expt_chroma('tracks-train.txt','tracks-test.txt',1,1000,4,0,1); toc
tic; [a3,c3,l3,m3] = do_expt_chroma('tracks-train-val.txt','tracks-val.txt',16,1000,4,0,1); toc
tic; [a3,c3,l3,m3] = do_expt_chroma('tracks-train.txt','tracks-test.txt',1,1000,1,0,1); toc
tic; [a3,c3,l3,m3] = do_expt_chroma('tracks-train.txt','tracks-test.txt',1,1000,2,0,1); toc
tic; [a3,c3,l3,m3] = do_expt_chroma('tracks-train.txt','tracks-test.txt',1,1000,3,0,1); toc
tic; [a3,c3,l3,m3] = do_expt_chroma('tracks-train.txt','tracks-test.txt',1,1000,1,1,1); toc
tic; [a3,c3,l3,m3] = do_expt_chroma('tracks-train.txt','tracks-test.txt',1,1000,2,1,1); toc
tic; [a3,c3,l3,m3] = do_expt_chroma('tracks-train.txt','tracks-test.txt',1,1000,3,1,1); toc
tic; [a3,c3,l3,m3] = do_expt_chroma('tracks-train.txt','tracks-test.txt',1,1000,4,1,1); toc
tic; [a3,c3,l3,m3] = do_expt_chroma('tracks-train.txt','tracks-test.txt',16,1000,1,0,1); toc
tic; [a3,c3,l3,m3] = do_expt_chroma('tracks-train.txt','tracks-test.txt',16,1000,4,0,1); toc
tic; [a3,c3,l3,m3] = do_expt_chroma('tracks-train.txt','tracks-test.txt',64,1000,4,0,1); toc
tic; [a3,c3,l3,m3] = do_expt_chroma('tracks-train.txt','tracks-test.txt',64,1000,1,0,1); toc
quit
%-- 6/14/07  3:08 PM --%
help calclistftrs
l1 = calclistftrs('../covers/list1.list','../covers/','.mp3','../chromftrs2/','.mat');
ls ../covers/All_Along_The_Watchtower/bob_dylan+Before_The_Flood_Disc_Two_+07-All_Along_the_Watchtower.mp3
ls -l ../covers/All_Along_The_Watchtower/bob_dylan+Before_The_Flood_Disc_Two_+07-All_Along_the_Watchtower.mp3
[d,sr] = mp3read('../covers/All_Along_The_Watchtower/bob_dylan+Before_The_Flood_Disc_Two_+07-All_Along_the_Watchtower.mp3',0,1,2);
l1 = calclistftrs('../covers/list1.list','../covers/','.mp3','../chromftrs2/','.mat');
size(d)
d2 = d/sqrt(mean(sum(d.^2))));
d2 = d/sqrt(mean(sum(d.^2)));
mean(sum(d.^2))
mean(sum(d2.^2))
l2 = calclistftrs('../covers/list2.list','../covers/','.mp3','../chromftrs2/','.mat');
help coverTestLists
[R,S,T,C] = coverTestLists(l1,l2,.5,2,1);
save cover80rslts
whos
clear d d2
save cover80rslts
[vv,xx] = max(R);
mean(xx==1:80)
diag(R)
diag(R)'
quit
