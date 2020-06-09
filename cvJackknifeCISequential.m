function [ CI, jackS ] = cvJackknifeCISequential( fullDataStatistic, dataFun, dataCell, alpha )
    %This function uses the jackknife to compute a confidence interval. Here the jackknife is performed by leaving out
    % one trial from a single dimension (for each class), so that the total number of leave-outs is the sum of trials across all dimensions
    %
    %fullDataStatistic is the statistic computed on all of the data.
    %
    %dataFun is a handle to the function that computes the statistic of
    %interest
    %
    %dataCell is nClasses { nDims { nObs x T } } cell that is input into dataFun. 
    %Matching data points are removed row-wise from dataCell to compute leave-one-out statistics for
    %the jackknife. 
    %
    % in sequential mode each entry of dataCell{.} is a cell array of size nDims x 1
    %
    % CI is 2 x nOuts (x time)
            
    nClasses = numel(dataCell);
    nDims = numel(dataCell{1});
    
    obsMat = cellfun(@(x) size(x, 1), cat(2, dataCell{:})'); % nClasses x nDims
    minObs = min(obsMat, [], 'all');
    assert(minObs >= 3, 'Minimum of 3 observations per dimension required for jackknife');
    
    nFoldsByDim = min(obsMat, [], 1);
    foldIdxByDim = cell(nDims, 1);
    for d = 1:nDims
        foldIdxByDim{d} = getFoldedIdx(obsMat(:, d), nFoldsByDim(d));
    end
    
    totalFolds = sum(nFoldsByDim);
    
    jackS = cell(nDims, 1);
    fprintf('Looping over dimensions');
    parfor d = 1:nDims
        nFolds = nFoldsByDim(d);
        jackS{d} = zeros(nFolds,size(fullDataStatistic, 1), size(fullDataStatistic, 2));
        deleteCell = dataCell;
        for f=1:nFolds
            for c=1:nClasses
                keep_idx = setdiff(1:obsMat(c, d), foldIdxByDim{d}{c, f});
                deleteCell{c}{d} = dataCell{c}{d}(keep_idx, :);
            end
            jackS{d}(f,:,:) = shiftdim(dataFun( deleteCell{:} ), -1); % nStats Time --> 1 x nStats x Time   
        end
        if mod(d, 10) == 0
            fprintf('%d / %d\n', d, nDims);
        end
    end
      
%     ps_norm = cell(nDims, 1);
%     for d = 1:nDims
%         ps_norm{d} = (nFoldsByDim(d)*shiftdim(fullDataStatistic, -1) - (nFoldsByDim(d)-1)*jackS{d}) / sqrt(nFoldsByDim(d)); 
%     end
%     
%     ps_norm = cat(1, ps_norm{:});
%     v_norm = var(ps_norm, [], 1);
%     
%     multiplier = norminv((1-alpha/2), 0, 1);
%     CI = [(shiftdim(fullDataStatistic, -1) - multiplier*sqrt(v_norm)); (shiftdim(fullDataStatistic, -1) + multiplier*sqrt(v_norm/totalFolds))];
    
    jackS = cat(1, jackS{:});
    ps = totalFolds*shiftdim(fullDataStatistic, -1) - (totalFolds-1)*jackS; % folds x nStats x time
    
    outliers_mask = any(abs(zscore(ps, 0, 1)) > 6, 2); % folds x 1 x time
    ps_clean = ps;
    ps_clean(outliers_mask) = NaN;
    v = var(ps_clean, [], 1, 'omitnan'); % 1 x nStats x time

    multiplier = norminv((1-alpha/2), 0, 1);
    CI = [(shiftdim(fullDataStatistic, -1) - multiplier*sqrt(v/totalFolds)); (shiftdim(fullDataStatistic, -1) + multiplier*sqrt(v/totalFolds))];
end

