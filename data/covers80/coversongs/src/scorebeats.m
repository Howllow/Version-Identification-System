function s = scorebeats(sbts,gbts)
% s = scorebeats(sbts,gbts)
%   Score candidate beat track sequence
%   sbts is system beat times, gbts is matrix of user-generated
%   ground-truth tapping.
% 2007-05-21 dpwe@ee.columbia.edu

bdiffs = [];

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

    disp([num2str(nhits),'/',num2str(mlb),' = ',num2str(pscore)]);
    
    avscore = avscore + pscore;
    tgtbts = tgtbts + length(gb);
    
  end

  avscore = avscore/ngts;
  
  disp(['avg score=', num2str(avscore),' tgtbts=',num2str(tgtbts)]);


