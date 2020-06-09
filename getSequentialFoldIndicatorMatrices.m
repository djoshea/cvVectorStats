function [bigFoldMatrices, smallFoldMatrices, nFoldsByDim, nObsBig, nObsSmall] = getSequentialFoldIndicatorMatrices( obsCounts, varargin )
    % An internal function used to split data into folds for cross-validation for use with sequentially recorded data. 
    % obsCounts is nClasses x nDims matrix of observation counts.
    % big  and small foldMatrices is nClasses x nDims { nFoldsByDim(d) x obsByDimByClass(c, d) }. 
    % If normalizeForAveraging is false, these are logical indicating whether to include a given observation,
    % If normalizeForAveraging is true, then they are numeric weights that when left multiplied, will average the included observations
    % 
    % Note that a separate set of crossvalidation folds are used for each dimension.
        
    p = inputParser();
    p.addParameter('normalizeForAveraging', false, @islogical);
    p.parse(varargin{:});
    normalizeForAveraging = p.Results.normalizeForAveraging;
    
    minObs = min(obsCounts, [], 'all');
    assert(minObs >= 2, 'Minimum of 2 observations per dimension required for corss-validation');
    
    [nClasses, nDims] = size(obsCounts);

    nFoldsByDim = min(obsCounts, [], 1);

    [bigFoldMatrices, smallFoldMatrices, nObsSmall, nObsBig] = deal(cell(nClasses, nDims));

    for c=1:nClasses
        for d = 1:nDims
            [smallFoldIdx, nPerFold] = getFoldedIdx(obsCounts(c, d), nFoldsByDim(d));
            n = obsCounts(c, d);
            nObsSmall{c,d} = nPerFold';
            nObsBig{c,d} = n - nObsSmall{c,d};
            bigFoldMatrices{c,d} = build_idx_each_row(nFoldsByDim(d), n, smallFoldIdx, normalizeForAveraging, true);
            smallFoldMatrices{c,d} = build_idx_each_row(nFoldsByDim(d), n, smallFoldIdx, normalizeForAveraging, false);
        end
    end 
end

function mat = build_idx_each_row(rows, cols, idx_each_row, row_normalized, idx_complement)
    if row_normalized
        mat = zeros(rows, cols);
        if idx_complement
            for r = 1:rows
                mat(r, :) = 1 / (cols - numel( idx_each_row{r}));
                mat(r, idx_each_row{r}) = 0;
            end
        else
            for r = 1:rows
                mat(r, idx_each_row{r}) = 1 ./ numel(idx_each_row{r});
            end
        end
    else 
        if idx_complement
            mat = true(rows, cols);
            for r = 1:rows
                mat(r, idx_each_row{r}) = false;
            end
        else
            mat = false(rows, cols);
            for r = 1:rows
                mat(r, idx_each_row{r}) = true;
            end
        end
    end   
end
        
    
    
        