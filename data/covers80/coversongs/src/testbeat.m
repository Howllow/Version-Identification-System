function [score, data, bdiffs] = testbeat(tightness, alpha, offs, data)
% [score, data] = testbeat(tightness, alpha, offs, data)
%   Run beat tracking over all the dev data
%   to allow tuning of parameters.
% 2006-09-13 dpwe@ee.columbia.edu  mirex06 beat

if nargin < 1
  tightness = 2.7;
end
if nargin < 2
  alpha = 0.8;
end
if nargin < 3
  offs = 0.015;  % optimized - people are 15ms late c/w beattracker
end

ntest = 20;

testpath = '../mirex06train';

tmean = 120;
tsd = .7;

sgsrate = 8000/32;

if nargin < 4 
  % need to recalc data matrix
  for i = 1:ntest
    % Load the audio
    [d,sr] = wavread(fullfile(testpath, ['train',num2str(i), '.wav']));
    [t,rxc,D,fmm,sgsrate] = tempo(d,sr,tmean,tsd);
    data.rxc{i} = rxc;
    data.fmm{i} = fmm;
    data.t{i} = t;
    data.sgsrate{i} = sgsrate;
  end
  % and reload the ground truth
  for i = 1:ntest
    data.tempo{i} = textread(fullfile(testpath, ['train',num2str(i), '-tempo.txt']));
    data.gtbeats{i} = textread(fullfile(testpath, ['train',num2str(i), '.txt']));
  end
else
  for i = 1:ntest
    [t,v] = tempo([],0,tmean,tsd,data.fmm{i});
    data.t{i} = t;
  end
end

% run beat tracking
for i = 1:ntest

  % always use higher tempo 3.0, .73, 0.015 = 0.522
  %tmpo = max(data.t{i}([1 2]));
  % .. or use preferred tempo 2.7, .8, 0.015 = 0.565
  tmpo = data.t{i}(1+(data.t{i}(3)<=.5));
  data.beats{i} = offs + beat(data.fmm{i},data.sgsrate{i},tmpo,[tightness alpha]);
  fprintf(1,'.');
end
fprintf(1,'\n');

% do scoring
score = 0;
nscores = 0;

bdiffs = [];

for i = 1:ntest
  
  gbts = data.gtbeats{i};
  sbts = data.beats{i};
  
  avscore = 0;
  tgtbts = 0;
  
  % for each ground truth stream
  ngts = size(gbts,1);
  for j = 1:ngts
    
    % Keep only the gt and sys for times >= 5s
    mintime = 5.0;
    sb = sbts(sbts >= mintime);
    gb = gbts(j, gbts(j,:) >= mintime);
    
    % What is the threshold?
    W = 0.2 * median(diff(gb));
    
    % closest correspondences
    db = repmat(gb, length(sb), 1) - repmat(sb', 1, length(gb));
    [mind,minx] = min(abs(db));
    
    nhits = sum(mind < W);
    
    % collect stats on near-misses
    bdiffs = [bdiffs, gb(mind < W) - sb(minx(mind < W))];
    
    mlb = max(length(sb), length(gb));
    pscore = nhits/mlb;

%    disp([num2str(nhits),'/',num2str(mlb),' = ',num2str(pscore)]);
    
    score = score + pscore;
    nscores = nscores + 1;
    
    avscore = avscore + pscore;
    tgtbts = tgtbts + length(gb);
    
  end

  avscore = avscore/ngts;
  
  disp(['tr',num2str(i,'%02d'),' avg score=', num2str(avscore),...
       ' tgtbts=',num2str(tgtbts)]);

end

score = score/nscores;

disp(['total score = ', num2str(score, 3)]);

% >> [sc, da] = testtempo(120,.7,da); 
% >> sc
% sc =
%     0.7685
