function [ CI, jackS ] = cvJackknifeCISequential( fullDataStatistic, dataFun, dataCell, alpha )
    %This function uses the jackknife to compute a confidence interval. 
    
    %fullDataStatistic is the statistic computed on all of the data.
    
    %dataFun is a handle to the function that computes the statistic of
    %interest
    
    %dataCell is nClasses { nDims { nObs x T } } cell that is input into dataFun. 
    %Matching data points are removed row-wise from dataCell to compute leave-one-out statistics for
    %the jackknife. 
    %
    % in sequential mode each entry of dataCell{.} is a cell array of size nDims x 1
            
    nClasses = numel(dataCell);
    nDims = numel(dataCell{1});
    
    obsMat = cellfun(@(x) size(x, 1), cat(2, dataCell{:})'); % nClasses x nDims
    minObs = min(obsMat, [], 'all');
    assert(minObs >= 3, 'Minimum of 3 observations per dimension required for jackknife');
    
    [~, smallFoldIndices, nFolds] = getSequentialFoldIndicatorMatrices(obsMat); % C x D 

    jackS = zeros(nFolds,size(fullDataStatistic, 2), size(fullDataStatistic, 3));
    for f=1:nFolds
        deleteCell = dataCell;
        for c=1:nClasses
            for d = 1:nDims
                idx_del = smallFoldIndices{c, d}(f);
                deleteCell{c}{d}(idx_del,:,:) = [];
            end
        end
        jackS(f,:,:) = dataFun( deleteCell{:} );            
    end

    ps = nFolds*fullDataStatistic - (nFolds-1)*jackS;
    v = var(ps);
    
    multiplier = norminv((1-alpha/2), 0, 1);
    CI = [(fullDataStatistic - multiplier*sqrt(v/nFolds)); (fullDataStatistic + multiplier*sqrt(v/nFolds))];
end

