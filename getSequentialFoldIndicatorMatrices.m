function [bigFoldMatrices, smallFoldIndices, nFolds] = getSequentialFoldIndicatorMatrices( obsCounts, varargin )
    % An internal function used to split data into folds for cross-validation for use with sequentially recorded data. 
    % obsCounts is nClasses x nDims matrix of observation counts.
    % foldMatrices is  nClasses x nDims { nFolds x obsByDimByClass(c, d) } logical indicating whether to include
    % a given observation from that class, dimension in a 
    % for computing the means
    
    p = inputParser();
    p.addParameter('normalizeForAveraging', false, @islogical);
    p.addParameter('maxFolds', NaN, @isscalar);
    p.parse(varargin{:});
    normalizeForAveraging = p.Results.normalizeForAveraging;
    maxFolds = p.Results.maxFolds;
    if isnan(maxFolds)
        maxFolds = 3 * max(obsCounts, [], 'all');
    end
    
    minObs = min(obsCounts, [], 'all');
    assert(minObs >= 2, 'Minimum of 2 observations per dimension required for corss-validation');
    
    [nClasses, nDims] = size(obsCounts);
    
    nFolds = lcm_multi(obsCounts(:), maxFolds);
    
    nFolds = min(nFolds, maxFolds);
    
    [bigFoldMatrices, smallFoldIndices] = deal(cell(nClasses, nDims));
    for c=1:nClasses
        for d = 1:nDims
            n = obsCounts(c, d);
            smallFoldIndices{c, d} = mod(uint32(0:nFolds-1)', n) + 1;
            if normalizeForAveraging
                bigFoldMatrices{c,d}    = build_striped_allExceptOneHot(nFolds, n, 1/(n-1));
            else
                bigFoldMatrices{c,d}    = build_striped_allExceptOneHot(nFolds, n, 1);
            end
        end
    end
end

function y = lcm_multi(vals, maxVal)
    y = vals(1);
    for i = 2:numel(vals)
        y = lcm(y, vals(i));
        if y > maxVal
            y = maxVal;
            return;
        end
    end
end

% function mat = build_striped_oneHot(rows, cols)
%     % build a rows x cols matrix consisting of stacked identity matrices with 1s set to value
%     
%     inds = repmat((1:cols:rows)', 1, cols) + (0:cols-1)*(rows+1);
%     [i, j] = ind2sub([rows cols], inds(inds<rows*cols));
%     mat = sparse(i, j, true, rows, cols);
%     
% end

function mat = build_striped_allExceptOneHot(rows, cols, value)
    % build a rows x cols matrix consisting of stacked identity matrices with 1s set to value
    
    inds = repmat((1:cols:rows)', 1, cols) + (0:cols-1)*(rows+1);
    mat = repmat(value, rows, cols);
    mat(inds(inds <= rows*cols)) = 0;
end